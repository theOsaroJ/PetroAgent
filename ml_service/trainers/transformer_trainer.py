import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class TabTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch, features]
        x = self.fc_in(x).unsqueeze(1)      # [batch, seq=1, d_model]
        x = self.transformer(x)             # [batch, seq=1, d_model]
        return self.fc_out(x[:, 0, :])      # [batch, 1]

def train_transformer(X, y, epochs=10):
    X_t = torch.tensor(X.values, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model   = TabTransformer(X_t.shape[1])
    loss_fn = nn.MSELoss()
    opt     = torch.optim.Adam(model.parameters())

    history = []
    for _ in range(epochs):
        total = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        history.append(total / len(loader))
    return history
