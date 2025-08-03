import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def train_neural(X, y, epochs=10):
    X_t = torch.tensor(X.values, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = nn.Sequential(
        nn.Linear(X_t.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters())

    history = []
    for _ in range(epochs):
        total = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        history.append(total / len(loader))
    return history
