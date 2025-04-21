import wandb
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from net import TSPDGraphTransformerNetwork
from utils_train import get_loss_function, prepare_data

def mape(pred, target):
    return torch.mean(torch.abs((target - pred) / target)) * 100

def train(config, data_file):
    run = wandb.init(project='tspd-tf-scaled', config=config)
    config = wandb.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, test_data, _ = prepare_data(
        f'data/{data_file}.h5',
        pos_encoding_dim=config.pos_encoding_dim,
        split_ratios=[0.8, 0.2, 0.0],
        k = config.k
    )

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    model = TSPDGraphTransformerNetwork(
        in_channels=config.pos_encoding_dim,
        hidden_channels=config.pos_encoding_dim,
        out_channels=config.pos_encoding_dim,
        heads=config.heads,
        beta=config.beta,
        dropout=config.dropout,
        normalization=config.normalization,
        num_gat_layers=config.num_gat_layers,
        activation=config.activation
    ).to(device)
    
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.initial_lr)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.initial_lr)
    elif config.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.initial_lr)
    elif config.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.initial_lr)
    
    criterion = get_loss_function(config.loss_function)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0)

    wandb.watch(model, log="all", log_freq=1)

    best_val_mape = float('inf')
    
    for epoch in range(config.epochs):
        total_loss = 0
        total_mape = 0
        for data in train_loader:
            data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_mape += mape(out, data.y.view(-1, 1)).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_mape = total_mape / len(train_loader)
        wandb.log({"epoch": epoch, "train_loss": avg_train_mape})

        model.eval() # Switch to eval mode
        with torch.no_grad():
            val_loss = 0
            val_mape = 0
            for data in test_loader:
                data.to(device)
                out = model(data)
                loss = criterion(out, data.y.view(-1, 1))
                val_loss += loss.item()
                val_mape += mape(out, data.y.view(-1, 1)).item()
            avg_val_loss = val_loss / len(test_loader)
            avg_val_mape = val_mape / len(test_loader)
            wandb.log({"epoch": epoch, "val_loss": avg_val_mape})

            if avg_val_mape < best_val_mape:
                best_val_mape = avg_val_mape
                torch.save(model.state_dict(), f"tf_{data_file}_alpha_{epoch}.pth")
        
        model.train() # Switch back to train mode
        scheduler.step()

        print(f'Epoch {epoch}, Train Loss: {avg_train_loss}, Train MAPE : {avg_train_mape}, Val Loss: {avg_val_loss}, Val MAPE: {avg_val_mape}')

    run.finish()


if __name__ == "__main__":
    config = {
    'pos_encoding_dim': 8,
    'batch_size': 16,
    'heads': 4,
    'dropout': 0,
    'initial_lr': 0.01,
    'normalization': 'graph_norm',
    'optimizer': 'adam',
    'activation': 'elu',
    'num_gat_layers': 4,
    'loss_function': 'mse',
    'epochs': 100, 
    'T_0': 128,
    'beta': False,
    'k': None
}
    data_file = "scaled_refined_uniform_384000"
    train(config, data_file)