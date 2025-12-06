from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_download import default_tickers
from feature_engineering import load_csv_ticker, merge_and_align_ticker_data, add_features


def build_base_dataset():
    output_path = Path(__file__).parent / "processed"
    output_path.mkdir(exist_ok=True)
    
    tickers = [t.replace('.US', '') for t in default_tickers]
    ticker_data = {ticker: load_csv_ticker(ticker) for ticker in tickers}
    aligned_data = merge_and_align_ticker_data(ticker_data)

    aligned_path = output_path / "aligned.csv"
    aligned_data.to_csv(aligned_path)
    
    enhanced_data = add_features(aligned_data)
    features_path = output_path / "features_raw.csv"
    enhanced_data.to_csv(features_path)

    df = enhanced_data.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df = df.sort_index()

    ticker_feature_map = {}
    for col in df.columns:
        if "_" not in col:
            continue
        ticker_symbol, feature_name = col.split("_", 1)
        ticker_feature_map.setdefault(ticker_symbol, {})[feature_name] = col

    all_tickers = sorted(ticker_feature_map.keys())

    excluded_features = {"ret_1d", "open", "high", "low"}
    common_features = None

    for ticker in all_tickers:
        usable = {f for f in ticker_feature_map[ticker] if f not in excluded_features}
        if common_features is None:
            common_features = usable
        else:
            common_features = common_features & usable

    common_features = sorted(common_features)

    model_ready_tickers = []
    for ticker in all_tickers:
        has_all = all(f in ticker_feature_map[ticker] for f in common_features)
        has_ret = "ret_1d" in ticker_feature_map[ticker]
        if has_all and has_ret:
            model_ready_tickers.append(ticker)

    normalized_df = df.copy()
    epsilon = 1e-8

    for ticker in model_ready_tickers:
        feature_columns = [ticker_feature_map[ticker][feat] for feat in common_features]
        subset = normalized_df[feature_columns]
        normalized_df[feature_columns] = (
            (subset - subset.mean()) /
            (subset.std(ddof=0) + epsilon)
        )

    norm_path = output_path / "features.csv"
    normalized_df.to_csv(norm_path)

    return normalized_df


def build_multi_dataset(df, window=60):
    if "date" in df.columns:
        df = df.set_index("date")
    df = df.sort_index()

    ticker_feature_map = {}
    for col in df.columns:
        if "_" not in col:
            continue
        ticker_symbol, feature_name = col.split("_", 1)
        if ticker_symbol not in ticker_feature_map:
            ticker_feature_map[ticker_symbol] = {}
        ticker_feature_map[ticker_symbol][feature_name] = col

    all_tickers = sorted(ticker_feature_map.keys())

    excluded_features = {"ret_1d", "open", "high", "low"}
    common_features = None
    
    for ticker in all_tickers:
        ticker_features = {f for f in ticker_feature_map[ticker].keys() 
                          if f not in excluded_features}
        if common_features is None:
            common_features = ticker_features
        else:
            common_features = common_features & ticker_features
    
    common_features = sorted(common_features)

    model_ready_tickers = []
    for ticker in all_tickers:
        has_all_features = all(feat in ticker_feature_map[ticker] 
                              for feat in common_features)
        has_returns = "ret_1d" in ticker_feature_map[ticker]
        
        if has_all_features and has_returns:
            model_ready_tickers.append(ticker)

    normalized_df = df.copy()
    epsilon = 1e-8
    
    for ticker in model_ready_tickers:
        feature_columns = [ticker_feature_map[ticker][feat] 
                          for feat in common_features]
        ticker_subset = normalized_df[feature_columns]
        normalized_df[feature_columns] = (
            (ticker_subset - ticker_subset.mean()) / 
            (ticker_subset.std(ddof=0) + epsilon)
        )

    return_columns = [ticker_feature_map[ticker]["ret_1d"] 
                     for ticker in model_ready_tickers]
    return_clip_lower = -0.20
    return_clip_upper = 0.20
    y_returns = df[return_columns].clip(
        lower=return_clip_lower, 
        upper=return_clip_upper
    ).values

    X_list = []
    for ticker in model_ready_tickers:
        ticker_columns = [ticker_feature_map[ticker][feat] 
                         for feat in common_features]
        ticker_features = normalized_df[ticker_columns].values
        ticker_features_expanded = ticker_features[:, np.newaxis, :]
        X_list.append(ticker_features_expanded)

    X_full = np.concatenate(X_list, axis=1)

    X_windows = []
    y_windows = []
    total_timesteps = X_full.shape[0]
    num_windows = total_timesteps - window

    for i in range(num_windows):
        window_start = i
        window_end = i + window
        X_window = X_full[window_start:window_end]
        X_window_transposed = np.transpose(X_window, (1, 0, 2))
        y_index = window_end
        y_value = y_returns[y_index]
        X_windows.append(X_window_transposed)
        y_windows.append(y_value)


    X_final = np.stack(X_windows).astype(np.float32)
    y_final = np.stack(y_windows).astype(np.float32)
    return X_final, y_final, model_ready_tickers, common_features

def save_multi_dataset():
    output_path = Path(__file__).parent / "processed"
    features_path = output_path / "features.csv"
    
    df = pd.read_csv(features_path, parse_dates=["date"])
    X_final, y_final, model_ready_tickers, common_features = build_multi_dataset(df, window=60)
    
    np.save(output_path / "multi_X.npy", X_final)
    np.save(output_path / "multi_y.npy", y_final)
    np.save(output_path / "multi_tickers.npy", np.array(model_ready_tickers))
    
    return X_final, y_final, model_ready_tickers, common_features


def split_by_date(df, train_start, train_end, val_end, test_end=None):
    train_df = df.loc[train_start:train_end]
    val_df = df.loc[train_end:val_end]
    test_df = df.loc[val_end:test_end] if test_end else df.loc[val_end:]
    return train_df, val_df, test_df


if __name__ == "__main__":
    print("Installing base dataset...")
    df = build_base_dataset()
    print("Base dataset built.")

    print("Building multi-ticker dataset...")
    X, y, tickers, features = save_multi_dataset()
    print("Done!")
    print("Shapes:", X.shape, y.shape)