import os, math, numpy as np, torch
from .base import train_val_split, regression_metrics
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class TabTransformer(nn.Module):
    def __init__(self, in_dim, d_model=128, nhead=8, depth=4, drop=0.1):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=drop, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
    def forward(self, x):
        # x: (B, F) -> treat each sample as a single token with dense embedding
        h = self.embed(x).unsqueeze(1)      # (B,1,d)
        h = self.encoder(h)                 # (B,1,d)
        return self.head(h.squeeze(1)).squeeze(-1)

def train_transformer(X, y, out_dir: str):
    Xtr, Xte, ytr, yte = train_val_split(X, y)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    yte = torch.tensor(yte, dtype=torch.float32)

    model = TabTransformer(in_dim=Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    lossf = nn.SmoothL1Loss()
    dl = DataLoader(TensorDataset(Xtr,ytr), batch_size=128, shuffle=True)

    best, bad, patience = math.inf, 0, 12
    for epoch in range(250):
        model.train()
        for xb,yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(Xte), yte).item()
        if v < best: best = v; bad = 0; best_state = model.state_dict()
        else: bad += 1
        if bad >= patience: break

    model.load_state_dict(best_state)
    with torch.no_grad():
        yp = model(Xte).cpu().numpy()
    metrics = regression_metrics(yte.cpu().numpy(), yp)
    model_path = os.path.join(out_dir, "transformer_model.pt")
    torch.save(model.state_dict(), model_path)
    return {"metrics": metrics, "y_true": yte.cpu().numpy().tolist(), "y_pred": yp.tolist(), "model_path": model_path}
