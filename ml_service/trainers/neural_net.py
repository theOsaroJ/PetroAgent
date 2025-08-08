import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch

def train(X, y):
    X_t = torch.tensor(X.values, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32).view(-1,1)
    ds = TensorDataset(X_t,y_t)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    model = nn.Sequential(
        nn.Linear(X.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,1)
    )
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(50):
        for xb,yb in dl:
            pred = model(xb)
            loss = loss_fn(pred,yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model
