import os
import numpy as np
import pandas as pd
import torch

from utils import plots

def evaluate(model, dataloader, target_cols, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device

    predictions = []
    target_labels = []

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y_hat = model(X).cpu()

            predictions.append(y_hat)
            target_labels.append(y)

    predictions = torch.cat(predictions, dim=0).numpy()
    target_labels = torch.cat(target_labels, dim=0).numpy()
    tickers = [c.replace("_ret_1d", "") for c in target_cols]

    target_labels_std, predictions_std, directional_accuracy  = plots.create_common_plots( predictions,target_labels, tickers, save_dir)

    df_metrics = pd.DataFrame({
        "ticker": tickers,
        "mse": ((predictions - target_labels) ** 2).mean(axis=0),
        "mae": np.abs(predictions - target_labels).mean(axis=0),
        "directional_accuracy": directional_accuracy,
        "pred_std": predictions_std,
        "true_std": target_labels_std
    })

    print(f"\n=== Global Metrics ===")
    print(f"MSE: {df_metrics['mse'].mean():.6f}")
    print(f"MAE: {df_metrics['mae'].mean():.6f}")
    print(f"Directional Accuracy: {df_metrics['directional_accuracy'].mean():.6f}")

    df_metrics.to_csv(f"{save_dir}/metrics.csv", index=False)
    print(f"Plots and metrics saved in: {save_dir}")
