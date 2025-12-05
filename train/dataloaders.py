import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

def train_val_test_split(df, train_start, train_end, val_end, test_end):
    train_df = df.loc[train_start:train_end]
    val_df = df.loc[train_end:val_end]
    test_df = df.loc[val_end:test_end]
    return train_df, val_df, test_df

class MultiAssetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MarketDataset(Dataset):
    def __init__(self, df, window_size=60, horizon=1, target_cols=None):
        self.df = df
        self.window_size = window_size
        self.horizon = horizon
        
        if target_cols is None:
            self.target_cols = [c for c in df.columns if c.endswith("_ret_1d")]
        else:
            self.target_cols = target_cols
            
        self.data = df.values.astype(np.float32)
        self.num_features = self.data.shape[1]
        self.num_targets = len(self.target_cols)
        self.target_idx = [df.columns.get_loc(c) for c in self.target_cols]

    def __len__(self):
        return len(self.df) - self.window_size - self.horizon

    def __getitem__(self, i):
        X = self.data[i:i + self.window_size]
        y_idx = i + self.window_size + self.horizon - 1
        y = self.data[y_idx, self.target_idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    


