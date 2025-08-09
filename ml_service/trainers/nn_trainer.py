import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class TorchRegressor:
    def __init__(self, model: nn.Module):
        self.model = model

    def fit(self, X, y, X_val=None, y_val=None, epochs=150, lr=1e-3, batch=256):
        device = torch.device("cpu")
        m = self.model.to(device)
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        if X_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

        opt = optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        best = float("inf"); best_state=None; patience=20; bad=0
        for ep in range(epochs):
            m.train()
            perm = torch.randperm(X.size(0))
            for i in range(0, X.size(0), batch):
                idx = perm[i:i+batch]
                xb = X[idx]; yb = y[idx]
                opt.zero_grad()
                pred = m(xb); loss = loss_fn(pred, yb)
                loss.backward(); opt.step()

            if X_val is not None:
                m.eval()
                with torch.no_grad():
                    vpred = m(X_val)
                    vl = loss_fn(vpred, y_val).item()
                if vl < best - 1e-6:
                    best = vl; best_state = {k:v.cpu().clone() for k,v in m.state_dict().items()}; bad=0
                else:
                    bad += 1
                    if bad >= patience:
                        break

        if best_state:
            m.load_state_dict(best_state)

    def predict(self, X):
        m = self.model
        m.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            return m(X).cpu().numpy()

def train_nn(X_train, y_train, X_test, y_test):
    d = X_train.shape[1]
    model = TorchRegressor(MLP(d))
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test, epochs=200)
    return model
