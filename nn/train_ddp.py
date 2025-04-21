import os
import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from net import TSPDGraphTransformerNetwork
from utils_train import get_loss_function, prepare_data

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # Set this to the correct address
    os.environ['MASTER_PORT'] = '12355'      # Set this to the correct port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, config, data_file):
    setup(rank, world_size)

    if rank == 0:
        wandb.init(config=config, project='tspd-tf')

    config = wandb.config if rank == 0 else config

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    print(f'Running on device: {device}, Current CUDA device: {torch.cuda.current_device()}, CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    print('-'*20)

    train_data, test_data, _ = prepare_data(
        f'data/{data_file}.h5',
        pos_encoding_dim=config.pos_encoding_dim,
        split_ratios=[0.8, 0.2, 0.0],
        device=device
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, num_replicas=world_size, rank=rank)

    train_loader = GeoDataLoader(train_data, batch_size=config['batch_size'], shuffle=False, num_workers=0, sampler=train_sampler)
    test_loader = GeoDataLoader(test_data, batch_size=config['batch_size'], shuffle=False, num_workers=0, sampler=test_sampler)

    model = TSPDGraphTransformerNetwork(
        in_channels=config['pos_encoding_dim'],
        hidden_channels=config['pos_encoding_dim'],
        out_channels=config['pos_encoding_dim'],
        heads=config['heads'],
        dropout=config['dropout'],
        mlp_hidden_layers=config['mlp_hidden_layers'],
        fill_value=config['fill_value'],
        normalization=config['normalization'],
        num_gat_layers=config['num_gat_layers'],
        activation=config['activation']
    ).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['initial_lr'])
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['initial_lr'])
    elif config['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config['initial_lr'])
    elif config['optimizer'] == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config['initial_lr'])
    
    criterion = get_loss_function(config['loss_function'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config["T_0"])

    if rank == 0:
        wandb.watch(model, log="all", log_freq=1)

    best_val_loss = float('inf')
    # patience = 10  # You can set the patience value as needed
    # epochs_no_improve = 0

    model.train()
    for epoch in range(200):
        total_loss = 0
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad()
        for data in train_loader:
            out = model(data)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        reduced_train_loss = torch.tensor(total_loss / len(train_loader)).to(device)
        dist.reduce(reduced_train_loss, dst=0, op=dist.ReduceOp.SUM)
        reduced_train_loss = reduced_train_loss.item() / world_size

        if rank == 0:
            wandb.log({"epoch": epoch, "train_loss": reduced_train_loss})

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for data in test_loader:
                out = model(data)
                loss = criterion(out, data.y.view(-1, 1))
                val_loss += loss.item()

            reduced_val_loss = torch.tensor(val_loss / len(test_loader)).to(device)
            dist.reduce(reduced_val_loss, dst=0, op=dist.ReduceOp.SUM)
            reduced_val_loss = reduced_val_loss.item() / world_size

            if rank == 0:
                wandb.log({"epoch": epoch, "val_loss": reduced_val_loss})

                if reduced_val_loss < best_val_loss:
                    # best_val_loss = reduced_val_loss
                    torch.save(model.state_dict(), f"tf_p2_{data_file}_{epoch}.pth")
                #     epochs_no_improve = 0
                # else:
                #     epochs_no_improve += 1

                # if epochs_no_improve >= patience:
                #     print(f'Early stopping at epoch {epoch}')
                #     break
        
        model.train()
        scheduler.step()

        if rank == 0:
            print(f'Epoch {epoch}, Train Loss: {reduced_train_loss}, Val Loss: {reduced_val_loss}')

    if rank == 0:
        wandb.finish()

    cleanup()
    

def main():
    device_ids = [0, 2, 3]
    world_size = len(device_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))

    config = {
        'pos_encoding_dim': 8,
        'batch_size': 64,
        'fill_value': "max",
        'heads':4,
        'dropout': 0,
        'initial_lr': 0.001,
        'normalization': 'graph_norm',
        'optimizer': 'sgd',
        'activation': 'elu',
        'num_gat_layers': 2,
        'loss_function': 'smooth_l1',
        'epochs': 100, 
        'T_0': 64
    }
    
    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)

if __name__ == "__main__":
    os.environ['NCCL_SHM_DISABLE'] = '1'
    print(f'CUDA version: {torch.version.cuda}, PyTorch version: {torch.__version__}, Is CUDA available: {torch.cuda.is_available()}')
    print('-'*20)
    main()
