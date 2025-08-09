import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .utils import metric_dict, plot_pred_vs_actual, plot_residuals
from .base_torch import EarlyStopper, tensors_from_xy, save_torch
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, in_features, hidden=[256,128,64], p=0.1):
        super().__init__()
        layers = []
        last = in_features
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(p)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_neural_net(X, y, save_dir, test_size=0.2, random_state=42, epochs=200, batch_size=64, lr=1e-3, hidden=None, dropout=0.1):
    hidden = hidden or [256,128,64]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    xtr, ttr = torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(y_train.values.reshape(-1,1), dtype=torch.float32)
    xva, tva = torch.tensor(X_val_s, dtype=torch.float32), torch.tensor(y_val.values.reshape(-1,1), dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(xtr, ttr), batch_size=batch_size, shuffle=True)

    model = MLP(X.shape[1], hidden=hidden, p=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    early = EarlyStopper(patience=20, min_delta=1e-5)

    for epoch in range(epochs):
        model.train()
        for xb, tb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, tb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(xva)
            val_loss = loss_fn(val_pred, tva).item()
        if early.step(val_loss):
            break

    with torch.no_grad():
        yp = model(xva).numpy().squeeze()
    metrics = metric_dict(y_val.values, yp)

    # save artifacts
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "model_mlp.pt")
    scaler_path = os.path.join(save_dir, "scaler_mlp.npy")
    torch.save(model.state_dict(), model_path)
    np.save(scaler_path, {"mean": scaler.mean_, "scale": scaler.scale_}, allow_pickle=True)

    p1 = os.path.join(save_dir, "pred_vs_actual_mlp.png")
    p2 = os.path.join(save_dir, "residuals_mlp.png")
    plot_pred_vs_actual(y_val.values, yp, p1)
    plot_residuals(y_val.values, yp, p2)

    return {
        "metrics": metrics,
        "artifacts": [p1, p2, scaler_path],
        "model_path": model_path,
        "notes": "PyTorch MLP with BN+Dropout and early stopping."
    }
