import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

raw_data_path = os.path.join(os.path.dirname(__file__), "raw")

# Load raw data for each ticker
def load_csv_ticker(ticker):
    path = os.path.join(raw_data_path, f"{ticker}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data csv file not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()

# Merge all tickers into 1 combined dataset
def merge_and_align_ticker_data(dfs):
    rename_columns = {
        ticker: df.rename(columns={col: f"{ticker}_{col}" for col in df.columns})
        for ticker, df in dfs.items()
    }
    combined = pd.concat(rename_columns.values(), axis=1).sort_index()
    return combined.ffill().bfill()

# Normalize prices
def normalize_features(df, start_date, end_date):
    training_data = df.loc[start_date:end_date]
    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    scaler = StandardScaler().fit(training_data.values)
    scaled = scaler.transform(df.values)
    scaled_df = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    return scaled_df, scaler

# Create features for models. The code for features is referenced from the course Machine Learning for Trading
def get_tickers(columns):
    tickers = set()
    for col in columns:
        if "_" in col:
            ticker = col.split("_")[0]
            tickers.add(ticker)
    return sorted(tickers)

def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(window).mean()
    loss = down.rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    line = fast_ema - slow_ema
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def bollinger(series, window=20, num_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    pct_b = (series - lower) / (upper - lower)
    return upper, lower, pct_b

def atr(high, low, close, window=14):
    prev = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev).abs(),
        (low - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def volume_z(volume, window=20):
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / std

def obv(close, volume):
    dir = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (dir * volume).fillna(0).cumsum()

def stochastic(high, low, close, k_window=14, d_window=3):
    ll = low.rolling(k_window).min()
    hh = high.rolling(k_window).max()
    k = 100 * (close - ll) / (hh - ll)
    d = k.ewm(span=d_window, adjust=False).mean()
    return k, d

def add_features(df):
    tickers = get_tickers(df.columns)
    frames = [df]

    for ticker in tickers:
        c_col = f"{ticker}_close"
        h_col = f"{ticker}_high"
        l_col = f"{ticker}_low"
        v_col = f"{ticker}_volume"

        if c_col not in df.columns:
            continue

        c = df[c_col]
        h = df.get(h_col)
        l = df.get(l_col)
        v = df.get(v_col)

        feats = {}
        feats[f"{ticker}_ret_1d"] = np.log(c / c.shift(1))
        feats[f"{ticker}_ret_5d"] = np.log(c / c.shift(5))
        feats[f"{ticker}_vol_20"] = feats[f"{ticker}_ret_1d"].rolling(20).std()
        feats[f"{ticker}_rsi_14"] = rsi(c)

        line, sig, hist = macd(c)
        feats[f"{ticker}_macd"] = line
        feats[f"{ticker}_macd_signal"] = sig
        feats[f"{ticker}_macd_hist"] = hist

        feats[f"{ticker}_ema_20"] = ema(c, span=20)
        feats[f"{ticker}_ema_50"] = ema(c, span=50)

        upper, lower, pct_b = bollinger(c)
        feats[f"{ticker}_boll_upper"] = upper
        feats[f"{ticker}_boll_lower"] = lower
        feats[f"{ticker}_boll_percent_b"] = pct_b

        feats[f"{ticker}_atr_14"] = atr(h, l, c, window=14)
        feats[f"{ticker}_volume_z"] = volume_z(v)
        feats[f"{ticker}_obv"] = obv(c, v)

        if h is not None and l is not None:
            k, d = stochastic(h, l, c)
            feats[f"{ticker}_stoch_k"] = k
            feats[f"{ticker}_stoch_d"] = d

        frames.append(pd.DataFrame(feats))

    out = pd.concat(frames, axis=1)
    out = out.ffill().bfill()
    return out