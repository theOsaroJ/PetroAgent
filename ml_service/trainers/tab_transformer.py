import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .utils import metric_dict, plot_pred_vs_actual, plot_residuals
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class TabTransformer(nn.Module):
    def __init__(self, in_features, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input = nn.Linear(in_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        # x: (B, F) -> (B, F, d_model) treat features as sequence positions
        z = self.input(x)              # (B, d_model)
        z = z.unsqueeze(1)             # (B, 1, d_model)
        z = self.encoder(z)            # (B, 1, d_model)
        out = self.head(z.squeeze(1))  # (B, 1)
        return out

def train_tab_transformer(X, y, save_dir, test_size=0.2, random_state=42, epochs=200, batch_size=64, lr=2e-4, d_model=128, nhead=8, num_layers=4, ff=256, dropout=0.1):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    xtr = torch.tensor(X_train_s, dtype=torch.float32)
    ttr = torch.tensor(y_train.values.reshape(-1,1), dtype=torch.float32)
    xva = torch.tensor(X_val_s, dtype=torch.float32)
    tva = torch.tensor(y_val.values.reshape(-1,1), dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(xtr, ttr), batch_size=batch_size, shuffle=True)

    model = TabTransformer(
        in_features=X.shape[1],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=ff,
        dropout=dropout
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    best = None
    bad = 0
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
        if best is None or val_loss < best - 1e-5:
            best = val_loss
            bad = 0
        else:
            bad += 1
            if bad >= 20:
                break

    with torch.no_grad():
        yp = model(xva).numpy().squeeze()

    metrics = metric_dict(y_val.values, yp)

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "model_tab_transformer.pt")
    torch.save(model.state_dict(), model_path)
    scaler_path = os.path.join(save_dir, "scaler_tab_transformer.npy")
    np.save(scaler_path, {"mean": scaler.mean_, "scale": scaler.scale_}, allow_pickle=True)

    p1 = os.path.join(save_dir, "pred_vs_actual_tab_transformer.png")
    p2 = os.path.join(save_dir, "residuals_tab_transformer.png")
    plot_pred_vs_actual(y_val.values, yp, p1)
    plot_residuals(y_val.values, yp, p2)

    return {
        "metrics": metrics,
        "artifacts": [p1, p2, scaler_path],
        "model_path": model_path,
        "notes": "Tabular transformer encoder (CPU)."
    }
