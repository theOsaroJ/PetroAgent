import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad = 0

    def step(self, val_loss):
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience

def tensors_from_xy(X, y):
    x = torch.tensor(X.values, dtype=torch.float32)
    t = torch.tensor(y.values.reshape(-1,1), dtype=torch.float32)
    return x, t

def save_torch(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
