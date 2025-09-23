# pre_train_fr.py
import argparse, os, torch, wandb
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from net_fr import (
    TSPDGraphTransformerNetworkFlyingRange,
    load_legacy_into_flying_range,
    freeze_legacy_and_enable_new,
)
from utils_train_fr import prepare_data_flying_range, get_loss_function

g_args = None  # filled from CLI in __main__

def mape(pred, target):
    return torch.mean(torch.abs((target - pred) / target.clamp(min=1e-12))) * 100

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train():
    run = wandb.init()
    cfg = wandb.config
    device = torch.device(g_args.device)

    # Data
    train_data, val_data, _ = prepare_data_flying_range(
        file_path=f"data/{g_args.data_file}.h5",
        pos_encoding_dim=8,
        split_ratios=(0.8, 0.2, 0.0),
        device=None,
        k=g_args.k,
    )
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader   = DataLoader(val_data,   batch_size=cfg.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # Model
    model = TSPDGraphTransformerNetworkFlyingRange(
        dropout=cfg.dropout,
        edge_dim=3,
        phi_layers=cfg.phi_layers,
        phi_hidden=cfg.phi_hidden,
        phi_activation=cfg.phi_activation,
        phi_leaky_slope=cfg.phi_leaky_slope,
        film_init=cfg.film_init,
        film_std=cfg.sigma_choices,
        edge3_init=cfg.edge3_init
    ).to(device)

    # === Load legacy and freeze legacy params ===
    assert os.path.exists(g_args.legacy_ckpt), f"legacy_ckpt not found: {g_args.legacy_ckpt}"
    load_legacy_into_flying_range(model, g_args.legacy_ckpt, device)
    freeze_legacy_and_enable_new(model)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg.initial_lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T0)
    criterion = get_loss_function(cfg.loss_function)

    wandb.watch(model, log="all", log_freq=50)
    wandb.log({"trainable_params": count_trainable_params(model)})

    best_val = float('inf')
    for epoch in range(g_args.epochs):
        # ---- Train ----
        model.train()
        tr_loss = tr_mape = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = criterion(out, batch.y.view(-1,1))
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tr_mape += mape(out, batch.y.view(-1,1)).item()

        # ---- Val ----
        model.eval()
        va_loss = va_mape = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y.view(-1,1))
                va_loss += loss.item()
                va_mape += mape(out, batch.y.view(-1,1)).item()

        ntr, nva = max(1, len(train_loader)), max(1, len(val_loader))
        log = {
            "epoch": epoch,
            "train_loss": tr_loss / ntr,
            "train_mape": tr_mape / ntr,
            "val_loss": va_loss / nva,
            "val_mape": va_mape / nva,
            "lr": scheduler.get_last_lr()[0],
        }
        wandb.log(log)

        if log["val_loss"] < best_val:
            best_val = log["val_loss"]

        scheduler.step()

        print(f"Epoch {epoch:03d} | Train {log['train_loss']:.6f}/{log['train_mape']:.2f} "
              f"| Val {log['val_loss']:.6f}/{log['val_mape']:.2f} | LR {log['lr']:.2e}")

    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", default="scaled_corrected_fr_12800", help="HDF5 file under data/ (without .h5)")
    ap.add_argument("--legacy_ckpt", default="../nn/trained/neural_cost_predictor.pth", help="path to legacy checkpoint (edge_dim=2)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--wandb_project", default="tspd-tf-flying-range-sweep")
    ap.add_argument("--count", type=int, default=200, help="number of sweep runs")

    args = ap.parse_args()
    g_args = args  # stash for train()

    # Sweep space
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            # φ-MLP
            "phi_layers":     {"values": [0, 1, 2]},        # 0 → φ≡1 → g(f)=f
            "phi_hidden":     {"values": [8, 16, 32, 64]},
            "phi_activation": {"values": ["relu", "leaky_relu", "elu"]},
            "phi_leaky_slope":{"values": [0.01]},

            # FiLM / edge init
            "sigma_choices":  {"values": [1e-4, 1e-3, 5e-3]},  # film_std
            "film_init":      {"values": ["zeros", "normal"]},
            "edge3_init":     {"values": ["default", "zero"]},

            # training knobs
            "batch_size": {"values": [16, 32, 64, 128, 256]},
            "dropout":    {"values": [0.0, 0.1, 0.2, 0.5]},
            "initial_lr": {"values": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]},
            "T0":         {"values": [32, 64, 128]},
            "loss_function": {"values": ["mse"]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=train, count=args.count)
