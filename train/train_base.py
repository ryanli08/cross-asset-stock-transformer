import yaml
import torch
import argparse
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from models.registry import get_model
from data.dataset import MarketDataset, split_by_date

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_train_loss(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)

def get_val_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)

def run_training(cfg):
    #load data
    df = pd.read_csv(cfg["data"]["path"], index_col=0, parse_dates=True)
    target_cols = [c for c in df.columns if c.endswith("_ret_1d")]

    train_df, val_df, test_df = split_by_date(
        df,
        cfg["data"]["train_start"],
        cfg["data"]["train_end"],
        cfg["data"]["val_end"],
        cfg["data"].get("test_end", None)
    )

    #create datasets and loaders
    window = cfg["training"]["window"]
    horizon = cfg["training"]["horizon"]
    
    train_ds = MarketDataset(train_df, window_size=window, horizon=horizon, target_cols=target_cols)
    val_ds = MarketDataset(val_df, window_size=window, horizon=horizon, target_cols=target_cols)
    test_ds = MarketDataset(test_df, window_size=window, horizon=horizon, target_cols=target_cols)

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    #model init
    num_features = train_ds.num_features
    num_targets = train_ds.num_targets
    
    model_cls = get_model(cfg["model"]["name"])
    model = model_cls(
        num_features=num_features,
        num_targets=num_targets,
        **{k: v for k, v in cfg["model"].items() if k != "name"}
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #training setup
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    best_val_loss = float("inf")
    patience = cfg["training"]["patience"]
    patience_ctr = 0

    #checkpoints
    save_dir = Path(cfg["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / f"{cfg['model']['name']}_best.pt"
    train_loss_history = []
    val_loss_history = []

    #training loop
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss = get_train_loss(model, train_loader, optimizer, criterion, device)
        val_loss = get_val_loss(model, val_loader, criterion, device)
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            torch.save({"model_state_dict": model.state_dict(), "config": cfg}, best_path)
            print("New best found")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Patience exceeded")
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss = get_val_loss(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_training(cfg)

