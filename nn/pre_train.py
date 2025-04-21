import wandb
import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from net import TSPDGraphTransformerNetwork
from utils_train import get_loss_function, prepare_data

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'pos_encoding_dim': {
            'values': [8]
        },
        'batch_size': {
            'values': [16, 32, 64, 128, 256]
        },
        'dropout': {
            'values': [0.0, 0.2, 0.5]
        },
        'initial_lr': {
            'values': [0.01, 0.001, 0.005, 0.0001]
        },
        'heads': {
            'values': [4]
        },
        'normalization': {
            'values': ['graph_norm']
        },
        'optimizer': {
            'values': ['adam']
        },
        'activation': {
            'values': ['elu']
        },
        'num_gat_layers': {
            'values': [4]
        },
        'loss_function': {
            'values': ['mse', 'huber', 'smooth_l1']
        },
        'T_0':{
            'values': [16, 32, 64, 128]
        },
        'beta': {
            'values': [False]
        },
        'readout':{
            'values': ['attention']
        }
    }
}
sweep_id = wandb.sweep(sweep_config, project="tspd-tf-scaled")

def mape(pred, target):
    return torch.mean(torch.abs((target - pred) / target)) * 100

def train(config=None):
    run = wandb.init(project='tspd-tf', config=config)
    config = wandb.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, val_data, _ = prepare_data(
        f'data/scaled_refined_uniform_38400.h5',
        pos_encoding_dim=config.pos_encoding_dim,
        split_ratios=[0.8, 0.2, 0.0]
    )

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    model = TSPDGraphTransformerNetwork(
        in_channels=config.pos_encoding_dim,
        hidden_channels=config.pos_encoding_dim,
        out_channels=config.pos_encoding_dim,
        heads=config.heads,
        beta=config.beta,
        dropout=config.dropout,
        normalization=config.normalization,
        num_gat_layers=config.num_gat_layers,
        activation=config.activation,
        readout_type=config.readout
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

    best_val_loss = float('inf')
    # patience = 10  # Number of epochs to wait before early stopping
    # no_improvement_epochs = 0
    
    for epoch in range(100):
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

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_mape = 0
            for data in val_loader:
                data.to(device)
                out = model(data)
                loss = criterion(out, data.y.view(-1, 1))
                val_loss += loss.item()
                val_mape += mape(out, data.y.view(-1, 1)).item()
            avg_val_loss = val_loss / len(val_loader)
            avg_val_mape = val_mape / len(val_loader)
            wandb.log({"epoch": epoch, "val_loss": avg_val_mape})

            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # no_improvement_epochs = 0
                # Save the best model if desired
                # torch.save(model.state_dict(), f"best_model_sweep_{run.id}_{epoch}.pth")
            # else:
            #     no_improvement_epochs += 1
            #     if no_improvement_epochs >= patience:
            #         print(f"Early stopping at epoch {epoch}. No improvement in validation loss for {patience} epochs.")
            #         break
        
        model.train()
        scheduler.step()

        print(f'Epoch {epoch}, Train Loss: {avg_train_loss}, Train MAPE: {avg_train_mape}, Val Loss: {avg_val_loss}, Val MAPE: {avg_val_mape}')

    run.finish()

if __name__ == "__main__":
    wandb.agent(sweep_id, train, count=100)