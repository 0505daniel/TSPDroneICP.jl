# train_fr.py
import argparse, os, torch, wandb
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from net_fr import (
    TSPDGraphTransformerNetworkFlyingRange,
    load_legacy_into_flying_range,
    freeze_legacy_and_enable_new,
)
from utils_train_fr import prepare_data_flying_range, get_loss_function


def mape(pred, target):
    return torch.mean(torch.abs((target - pred) / target.clamp(min=1e-12))) * 100

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--data_file", type=str, default="scaled_corrected_fr_128000", help="HDF5 file under data/ (without .h5)")
    ap.add_argument("--k", type=int, default=None, help="KNN degree (None = dense graph)")
    # Legacy
    ap.add_argument("--legacy_ckpt", type=str, default="../nn/trained/neural_cost_predictor.pth")
    # Training
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--initial_lr", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--T0", type=int, default=128)
    ap.add_argument("--loss_function", default="mse", choices=["mse", "mae", "huber", "smooth_l1"])
    ap.add_argument("--grad_clip", type=float, default=0.0, help="0 disables clipping")
    ap.add_argument("--seed", type=int, default=42)
    # φ-MLP / FiLM / edge init
    ap.add_argument("--phi_layers", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--phi_hidden", type=int, default=64)
    ap.add_argument("--phi_activation", default="leaky_relu", choices=["relu", "leaky_relu", "elu"])
    ap.add_argument("--phi_leaky_slope", type=float, default=0.01)
    ap.add_argument("--film_init", default="normal", choices=["zeros", "normal"])
    ap.add_argument("--film_std", type=float, default=0.0001)
    ap.add_argument("--edge3_init", default="default", choices=["default", "zero"])
    # Logging / Saving
    ap.add_argument("--wandb_project", default="tspd-tf-flying-range")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--save_dir", default="checkpoints")
    ap.add_argument("--save_best_name", default=None)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # Data
    train_data, val_data, _ = prepare_data_flying_range(
        file_path=f"data/{args.data_file}.h5",
        pos_encoding_dim=8,
        split_ratios=(0.8, 0.2, 0.0),
        device=None,
        k=args.k,
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=4)
    val_loader   = DataLoader(val_data,   batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, num_workers=4)

    # Model
    model = TSPDGraphTransformerNetworkFlyingRange(
        dropout=args.dropout,
        edge_dim=3,
        phi_layers=args.phi_layers,
        phi_hidden=args.phi_hidden,
        phi_activation=args.phi_activation,
        phi_leaky_slope=args.phi_leaky_slope,
        film_init=args.film_init,
        film_std=args.film_std,
        edge3_init=args.edge3_init,
    ).to(device)

    # === Load legacy and freeze legacy params ===
    assert os.path.exists(args.legacy_ckpt), f"legacy_ckpt not found: {args.legacy_ckpt}"
    load_legacy_into_flying_range(model, args.legacy_ckpt, device)
    freeze_legacy_and_enable_new(model)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.initial_lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T0)
    criterion = get_loss_function(args.loss_function)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        wandb.watch(model, log="all", log_freq=50)
        wandb.log({"trainable_params": count_trainable_params(model)})

    import os as _os
    _os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt = args.save_best_name or f"fr_main_{args.data_file}_best.pth"
    best_ckpt = _os.path.join(args.save_dir, best_ckpt)

    print(f"Trainable params: {count_trainable_params(model):,}")

    best_val = float("inf")
    for epoch in range(args.epochs):
        # ---- Train ----
        model.train()
        tr_loss = tr_mape = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = criterion(out, batch.y.view(-1, 1))
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                               max_norm=args.grad_clip)
            optimizer.step()
            tr_loss += loss.item()
            tr_mape += mape(out, batch.y.view(-1, 1)).item()

        # ---- Val ----
        model.eval()
        va_loss = va_mape = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y.view(-1, 1))
                va_loss += loss.item()
                va_mape += mape(out, batch.y.view(-1, 1)).item()

        ntr, nva = max(1, len(train_loader)), max(1, len(val_loader))
        log = {
            "epoch": epoch,
            "train_loss": tr_loss / ntr,
            "train_mape": tr_mape / ntr,
            "val_loss": va_loss / nva,
            "val_mape": va_mape / nva,
            "lr": scheduler.get_last_lr()[0],
        }
        if use_wandb:
            wandb.log(log)

        # Save best
        if log["val_loss"] < best_val:
            best_val = log["val_loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "epoch": epoch,
                "val_loss": best_val,
            }, best_ckpt)

        scheduler.step()

        print(f"Epoch {epoch:03d} | Train {log['train_loss']:.6f}/{log['train_mape']:.2f} "
              f"| Val {log['val_loss']:.6f}/{log['val_mape']:.2f} | LR {log['lr']:.2e}")

    if use_wandb:
        wandb.finish()

    print(f"Best val_loss: {best_val:.6f}")
    print(f"Saved best checkpoint to: {best_ckpt}")


if __name__ == "__main__":
    main()
