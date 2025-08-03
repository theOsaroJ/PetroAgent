import os, torch
import joblib
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

class Net(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

def train_neural(X: pd.DataFrame, y: pd.Series):
    X_t = torch.tensor(X.values, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = Net(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(50):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    path = f"{MODEL_DIR}/nn.pt"
    torch.save(model.state_dict(), path)
    with torch.no_grad():
        score = 1 - loss_fn(model(X_t), y_t).item()/torch.var(y_t).item()
    return {"model_path": path, "train_score": score}

def predict_neural(df: pd.DataFrame):
    model = Net(df.shape[1])
    model.load_state_dict(torch.load(f"{MODEL_DIR}/nn.pt"))
    with torch.no_grad():
        preds = model(torch.tensor(df.values, dtype=torch.float32)).squeeze().tolist()
    return {"predictions": preds}
