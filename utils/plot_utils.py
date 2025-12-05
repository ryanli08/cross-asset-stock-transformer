from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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

def std_comparison_plot(predictions, target_labels, tickers, save_dir):
    save_dir = Path(save_dir)
    file_name = save_dir / "std_comparison.png"
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html
    plt.figure(figsize=(10, 5))
    plt.title("Std Comparison (Collapse Check)")
    x = np.arange(len(tickers))
    plt.bar(x - 0.3, target_labels.std(axis=0), width=0.3, label="Actual Std")
    plt.bar(x + 0.3, predictions.std(axis=0), width=0.3, label="Predicted Std")
    step = max(1, len(tickers) // 40)
    plt.xticks(x[::step], tickers[::step], rotation=70, fontsize=7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

    return predictions.std(axis=0), target_labels.std(axis=0)

def directional_accuracy_plot(predictions, target_labels, tickers, save_dir):
    save_dir = Path(save_dir)
    file_name = save_dir / "directional_accuracy.png"
    directional_accuracy = []
    for i in range(len(tickers)):
        target_labels_up = target_labels[:, i] > 0
        predictions_up   = predictions[:, i] > 0
        directional_accuracy.append((target_labels_up == predictions_up).mean())

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

    return directional_accuracy

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



    
