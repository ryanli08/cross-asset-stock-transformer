from pathlib import Path
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def _calc_directional_accuracy(predictions, target_labels, tickers):
    directional_accuracy = []
    for i in range(len(tickers)):
        target_labels_up = target_labels[:, i] > 0
        predictions_up   = predictions[:, i] > 0
        directional_accuracy.append((target_labels_up == predictions_up).mean())
    return directional_accuracy

def _calc_attention_cross_norm(attention):
    attention_cross = attention.copy()
    np.fill_diagonal(attention_cross, 0)

    row_sums = attention_cross.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return  attention_cross / row_sums

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
    plt.hist(target_labels.flatten(), bins=80, alpha=0.5, label="Actual", density=True)
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

def attention_heatmap(attention, tickers, title, save_path):
    plt.figure(figsize=(12, 10))
    plt.title(title, fontsize=16)
    # https://seaborn.pydata.org/generated/seaborn.heatmap.html
    sns.heatmap(
        attention,
        xticklabels=tickers,
        yticklabels=tickers,
        cmap="viridis",
        cbar_kws={"label": "Attention Weight"},
        square=True,
    )
    plt.xlabel("Attended To (Keys)", fontsize=12)
    plt.ylabel("Attending From (Queries)", fontsize=12)
    plt.xticks(rotation=70, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def head_heatmaps(attention_heads, tickers, save_dir, title_prefix):
    H = attention_heads.shape[0]
    for h in range(H):
        title = f"{title_prefix} — Head {h}"
        path = save_dir / f"real_cross_attention_head{h}.png"
        attention_heatmap(attention_heads[h], tickers, title, path)

def stock_sector_attention_heatmap(attention, tickers, save_dir):
    save_dir = Path(save_dir)
    file_name = save_dir / "real_stock_to_sector_attention.png"
    sector_etfs = {"XLE", "XLF", "XLK", "XLY"} 
    sector_indices = [i for i, t in enumerate(tickers) if t in sector_etfs]
    stock_indices = [i for i, t in enumerate(tickers) if t not in sector_etfs]

    if not sector_indices or not stock_indices:
        print("Skipping due to missing data")
        return

    if len(stock_indices) > 30:
        np.random.seed(42)
        stock_indices = np.random.choice(stock_indices, 35, replace=False).tolist()

    stock_to_sector = attention[np.ix_(stock_indices, sector_indices)]
    stock_names = [tickers[i] for i in stock_indices]
    sector_names = [tickers[i] for i in sector_indices]

    plt.figure(figsize=(8, 12))
    plt.title("Stock-to-Sector Attention", fontsize=16)
    # https://seaborn.pydata.org/generated/seaborn.heatmap.html
    sns.heatmap(
        stock_to_sector,
        xticklabels=sector_names,
        yticklabels=stock_names,
        cmap="viridis",
        cbar_kws={"label": "Attention Weight"},
        annot=True,
        fmt=".3f",
    )
    plt.xlabel("Sector ETFs", fontsize=12)
    plt.ylabel("Individual Stocks", fontsize=12)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close()

def clustered_attention(attention, tickers, save_dir):
    save_dir = Path(save_dir)
    file_name = save_dir / "real_cross_attention_clustered.png"
    # https://seaborn.pydata.org/generated/seaborn.clustermap.html
    cg = sns.clustermap(
        attention,
        row_cluster=True,
        col_cluster=True,
        row_linkage=None,
        col_linkage=None,
        xticklabels=tickers,
        yticklabels=tickers,
        cmap="viridis",
        figsize=(12, 12),
    )
    cg.figure.suptitle("Clustered Cross-Asset Attention", fontsize=16)
    cg.figure.tight_layout()
    cg.figure.subplots_adjust(top=0.93)
    cg.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close(cg.figure)

def create_common_plots(predictions, target_labels, tickers, save_dir):
    scatter_plot(predictions, target_labels, save_dir)
    error_histogram_plot(predictions, target_labels, save_dir)

    target_labels_std = target_labels.std(axis=0)
    predictions_std = predictions.std(axis=0)
    std_comparison_plot(predictions_std, target_labels_std, tickers, save_dir)
    
    directional_accuracy = _calc_directional_accuracy(predictions, target_labels, tickers)
    directional_accuracy_plot(directional_accuracy, tickers, save_dir)
    
    distribution_comparison_plot(predictions, target_labels, save_dir)
    
    timeseries_plot(predictions, target_labels, tickers, save_dir)

    return target_labels_std, predictions_std, directional_accuracy

def create_attention_plots(avg_attention, per_headwise_attention, tickers, save_dir):
    save_dir = Path(save_dir)
    
    attention_heatmap(
        avg_attention,
        tickers,
        "Real Cross-Asset Attention (Last Layer, All Heads Averaged)",
        save_dir / "real_cross_attention_full.png",
    )

    cross_attention_norm = _calc_attention_cross_norm(avg_attention)

    stock_sector_attention_heatmap(
        cross_attention_norm,
        tickers,
        save_dir,
    )

    head_heatmaps(
        per_headwise_attention,
        tickers,
        save_dir,
        "Real Cross-Asset Attention (Last Layer, Per Head)",
    )

    clustered_attention(
        cross_attention_norm,
        tickers,
        save_dir,
    )
