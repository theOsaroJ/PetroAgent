from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn, torch.optim as optim, torch

class Trans(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.fc_in = nn.Linear(d_in, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_out = nn.Linear(64,1)
    def forward(self, x):
        x = self.fc_in(x).unsqueeze(1)
        x = self.trans(x).squeeze(1)
        return self.fc_out(x)

def train(X, y):
    X_t = torch.tensor(X.values, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32).view(-1,1)
    ds = TensorDataset(X_t,y_t)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    model = Trans(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(50):
        for xb,yb in dl:
            pred = model(xb)
            loss = loss_fn(pred,yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model
