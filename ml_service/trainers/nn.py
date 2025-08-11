import os, joblib, math, numpy as np
from .base import train_val_split, regression_metrics
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[256,128,64], drop=0.2):
        super().__init__()
        layers=[]
        d=in_dim
        for h in hidden:
            layers += [nn.Linear(d,h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(drop)]
            d=h
        layers += [nn.Linear(d,1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

def train_mlp(X, y, out_dir: str):
    Xtr, Xte, ytr, yte = train_val_split(X, y)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    yte = torch.tensor(yte, dtype=torch.float32)

    model = MLP(in_dim=Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.SmoothL1Loss()
    dl = DataLoader(TensorDataset(Xtr,ytr), batch_size=128, shuffle=True)

    best_loss = math.inf
    patience, bad = 10, 0
    for epoch in range(200):
        model.train()
        for xb,yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
        # val
        model.eval()
        with torch.no_grad():
            val_loss = lossf(model(Xte), yte).item()
        if val_loss < best_loss: best_loss = val_loss; bad = 0; best_state = model.state_dict()
        else: bad += 1
        if bad >= patience: break

    model.load_state_dict(best_state)
    with torch.no_grad():
        yp = model(Xte).cpu().numpy()
    metrics = regression_metrics(yte.cpu().numpy(), yp)
    model_path = os.path.join(out_dir, "nn_model.pt")
    torch.save(model.state_dict(), model_path)
    return {"metrics": metrics, "y_true": yte.cpu().numpy().tolist(), "y_pred": yp.tolist(), "model_path": model_path}
