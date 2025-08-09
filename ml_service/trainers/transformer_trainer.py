import torch
import torch.nn as nn
import torch.optim as optim

class TinyTabTransformer(nn.Module):
    def __init__(self, d_in: int, d_model: int = 128, nhead: int = 8, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, d]
        h = self.input(x)                  # [B, d_model]
        h = h.unsqueeze(1)                 # [B, 1, d_model] (single token)
        h = self.encoder(h)                # [B, 1, d_model]
        y = self.out(h.squeeze(1)).squeeze(-1)
        return y

class TorchRegressor:
    def __init__(self, model: nn.Module):
        self.model = model

    def fit(self, X, y, X_val=None, y_val=None, epochs=200, lr=1e-3, batch=256):
        device = torch.device("cpu")
        m = self.model.to(device)
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        if X_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

        opt = optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        best = float("inf"); best_state=None; patience=25; bad=0
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

def train_transformer(X_train, y_train, X_val, y_val):
    d = X_train.shape[1]
    model = TorchRegressor(TinyTabTransformer(d))
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, epochs=300)
    return model
