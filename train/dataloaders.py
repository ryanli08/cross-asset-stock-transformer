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


