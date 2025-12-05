import os
import numpy as np
import pandas as pd
import torch

from utils import plots

def _calc_directional_accuracy(predictions, target_labels, tickers):
    directional_accuracy = []
    for i in range(len(tickers)):
        target_labels_up = target_labels[:, i] > 0
        predictions_up   = predictions[:, i] > 0
        directional_accuracy.append((target_labels_up == predictions_up).mean())
    return directional_accuracy

def evaluate(model, dataloader, target_cols, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device

    model.eval()
    predictions = []
    target_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y_hat = model(X).cpu()

            predictions.append(y_hat)
            target_labels.append(y)

    predictions = torch.cat(predictions, dim=0).numpy()
    target_labels = torch.cat(target_labels, dim=0).numpy()
    tickers = [c.replace("_ret_1d", "") for c in target_cols]

    plots.scatter_plot(predictions, target_labels, save_dir)

    plots.error_histogram_plot(predictions, target_labels, save_dir)
    
    target_labels_std = target_labels.std(axis=0)
    predictions_std = predictions.std(axis=0)
    plots.std_comparison_plot(predictions_std, target_labels_std, tickers, save_dir)
    
    directional_accuracy = _calc_directional_accuracy(predictions, target_labels, tickers)
    plots.directional_accuracy_plot(directional_accuracy, tickers, save_dir)
    
    plots.distribution_comparison_plot(predictions, target_labels, save_dir)
    
    plots.timeseries_plot(predictions, target_labels, tickers, save_dir)

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
