import argparse
import numpy as np
import pandas as pd
import sys
import torch
import yaml


from pathlib import Path
from torch.utils.data import DataLoader


sys.path.insert(0, str(Path(__file__).parent.parent))

from models.registry import get_model
from train.dataloaders import MultiAssetDataset
from utils import plots

def _load_config(path):
  with open(path, "r") as f:
    return yaml.safe_load(f)

def _run_evaluation(config_path):
    cfg = _load_config(config_path)
    model_name = cfg["model"]["name"]
    save_dir = Path(cfg["training"]["save_dir"])
    checkpoint_path =  save_dir / f"{model_name}_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    x_path = cfg["data"]["x_path"]
    y_path = cfg["data"]["y_path"]
    state_dict =  checkpoint["model_state_dict"]
    # https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load
    X = np.load(x_path)
    y = np.load(y_path)
    
    N, A, W, F = X.shape

    val_ratio = cfg["data"]["val_ratio"]
    train_ratio = cfg["data"]["train_ratio"]
    val_end  = int(N * (val_ratio + train_ratio))

    X_test = X[val_end:]
    y_test = y[val_end:]
    print(f"Test split shape: {X_test.shape}")

    test_ds = MultiAssetDataset(X_test, y_test) 
    # https://docs.pytorch.org/docs/stable/data.html#module-torch.utils.data
    test_loader = DataLoader(test_ds, batch_size=256)
    model_class = get_model(model_name)
    model_params = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = model_class(
      num_features=F,
      num_assets=A,
      **model_params,
    )
    model.load_state_dict(state_dict)
    print("Model loaded successfully")

    predictions_list = []
    target_labels_list = []

    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
          y_hat = model(X)
          predictions_list.append(y_hat)
          target_labels_list.append(y)

    # https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat
    predictions = torch.cat(predictions_list, dim=0).numpy()
    target_labels = torch.cat(target_labels_list, dim=0).numpy()

    results_dir = Path("results") / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    tickers = np.load(cfg["data"]["tickers_path"])

    target_labels_std, predictions_std, directional_accuracy  = plots.create_common_plots( predictions,target_labels, tickers, results_dir)

    mse = np.mean((predictions - target_labels) ** 2)
    # https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html#numpy-corrcoef
    corr = np.mean([np.corrcoef(target_labels[:, i], predictions[:, i])[0, 1] for i in range(A)])
    print(f"\n=== Global Metrics ===")
    print(f"MSE:                  {mse:.6f}")
    print(f"Mean Corr:            {corr:.4f}")
    print(f"Directional Accuracy: {np.mean(directional_accuracy):.4f}")
    print(f"Plots saved in: {results_dir}")

    df_metrics = pd.DataFrame({
        "ticker": tickers,
        "mse": mse,
        "mae": np.abs(predictions - target_labels).mean(axis=0),
        "directional_accuracy": directional_accuracy,
        "pred_std": predictions_std,
        "true_std": target_labels_std
    })
    df_metrics.to_csv(results_dir / "metrics.csv", index=False)


if __name__ == "__main__":
  # https://docs.python.org/3/howto/argparse.html
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, required=True)
  args = parser.parse_args()

  _run_evaluation(args.config)