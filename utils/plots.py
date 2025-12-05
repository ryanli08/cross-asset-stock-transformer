from pathlib import Path
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def loss_curves_plot(train_loss_history, val_loss_history, save_dir):
    save_dir = Path(save_dir)
    file_name = save_dir / "loss_curves.png"
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
    plt.figure(figsize=(8, 5))
    plt.title("Training and Validation Loss Curves")
    plt.plot(train_loss_history, label="Training Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def scatter_plot(predictions, target_labels, save_dir):
    save_dir = Path(save_dir)
    file_name = save_dir/ "scatter.png"
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html
    plt.figure(figsize=(8, 8))
    plt.title("Scatter: Actual vs Predicted (All Assets)")
    plt.scatter(target_labels.flatten(), predictions.flatten(), s=8, alpha=0.3)
    plt.xlabel("Actual Returns")
    plt.ylabel("Predicted Returns")
    plt.axline((0,0), slope=1, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def error_histogram_plot(predictions, target_labels, save_dir):
    save_dir = Path(save_dir)
    file_name = save_dir / "error_histogram.png"
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html
    plt.figure(figsize=(7,5))
    plt.title("Prediction Error Distribution (All Assets)")
    plt.hist(predictions.flatten() - target_labels.flatten(), bins=60, alpha=0.7)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def std_comparison_plot(predictions_std, target_labels_std, tickers, save_dir):
    save_dir = Path(save_dir)
    file_name = save_dir / "std_comparison.png"
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html
    plt.figure(figsize=(10, 5))
    plt.title("Std Comparison (Collapse Check)")
    x = np.arange(len(tickers))
    plt.bar(x - 0.3, target_labels_std, width=0.3, label="Actual Std")
    plt.bar(x + 0.3, predictions_std, width=0.3, label="Predicted Std")
    step = max(1, len(tickers) // 40)
    plt.xticks(x[::step], tickers[::step], rotation=70, fontsize=7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def directional_accuracy_plot(directional_accuracy, tickers, save_dir):
    save_dir = Path(save_dir)
    file_name = save_dir / "directional_accuracy.png"
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html
    plt.figure(figsize=(12, 5))
    plt.title("Directional Accuracy Per Ticker")
    plt.bar(tickers, directional_accuracy)
    step = max(1, len(tickers) // 40)
    x = np.arange(len(tickers))
    plt.xticks(x[::step], tickers[::step], rotation=70, fontsize=7)
    plt.ylim(0.4, 0.7)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def distribution_comparison_plot(predictions, target_labels, save_dir):
    save_dir = Path(save_dir)
    file_name = save_dir / "distribution_comparison.png"
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html
    plt.figure(figsize=(7, 5))
    plt.title("Distribution: Actual vs Predicted Returns")
    plt.hist(target_labels.flattent(), bins=80, alpha=0.5, label="Actual", density=True)
    plt.hist(predictions.flatten(), bins=80, alpha=0.5, label="Predicted", density=True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def timeseries_plot(predictions, target_labels, tickers, save_dir):
    try:
        spy_idx = tickers.index("AAPL")
    except:
        spy_idx = 0
    ticker = tickers[spy_idx]

    save_dir = Path(save_dir)
    file_name = save_dir / f"timeseries_{ticker}.png"

    plt.figure(figsize=(12, 4))
    plt.plot(target_labels[:, spy_idx], label="Actual", alpha=0.8)

    # https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.ndimage.uniform_filter1d.html
    window_size = 20
    target_labels_ma = uniform_filter1d(target_labels[:, spy_idx], window_size, mode='nearest')
    plt.plot(target_labels_ma, label=f"Actual (Uniform MA - {window_size})", color="green")

    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
    target_labels_casual_ma = pd.Series(target_labels[:, spy_idx]).rolling(window_size, min_periods=1).mean().values 
    plt.plot(target_labels_casual_ma, label=f"Actual (Causal Rolling MA - {window_size})", color="purple")
    plt.plot(predictions[:, spy_idx], label="Predicted", alpha=0.8, color="orange")
    plt.title(f"{ticker} — Time Series Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

    
