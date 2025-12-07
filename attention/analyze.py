
import argparse
import numpy as np
import pandas as pd
import torch
import sys
import yaml

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.registry import get_model
from utils import plots
from analyzer import Analyzer

def _load_config(path):
  with open(path, "r") as f:
    return yaml.safe_load(f)

def _run_analysis(config_path, num_samples):
    cfg = _load_config(config_path)
    model_name = cfg["model"]["name"]
    save_dir = Path(cfg["training"]["save_dir"])
    checkpoint_path =  save_dir / f"{model_name}_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict =  checkpoint["model_state_dict"]
    x_path = cfg["data"]["x_path"]
    y_path = cfg["data"]["y_path"]
    tickers_path =  cfg["data"]["tickers_path"] 

    X = np.load(x_path)
    y = np.load(y_path)
    N, A, W, F = X.shape

    tickers = np.load(tickers_path).tolist()
    print(f"Data: {X.shape}, Tickers: {len(tickers)}")

    model_class = get_model(model_name)
    model_params = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = model_class(
        num_features=F,
        num_assets=A,
        **model_params,
    )
    model.load_state_dict(state_dict)
    model.eval()
    print(" Model restored.")

    analyzer = Analyzer(model)
    val_ratio = cfg["data"]["val_ratio"]
    train_ratio = cfg["data"]["train_ratio"]
    val_end = int(N * (train_ratio + val_ratio))
    X_test = X[val_end : val_end + num_samples]
    X_tensor = torch.FloatTensor(X_test)
    avg_attention = analyzer.averaged_attention(X_tensor)
    per_headwise_attention = analyzer.headwise_attention(X_tensor)
    print(f"Real cross-attention matrix shape (averaged): {avg_attention.shape}")
    print(f"Per-head attention shape: {per_headwise_attention.shape}")
    print(f"Number of cross-encoder layers: {len(model.cross_asset_encoder.layers)}")

    save_dir = Path("results") / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    plots.create_attention_plots(avg_attention, per_headwise_attention, tickers, save_dir)

if __name__ == "__main__":
  # https://docs.python.org/3/howto/argparse.html
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, required=True)
  parser.add_argument("--num_samples", type=int, default=100)
  args = parser.parse_args()

  _run_analysis(args.config, args.num_samples)
