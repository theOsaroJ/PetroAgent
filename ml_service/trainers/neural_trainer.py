import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_neural(data_path: str, epochs: int = 10):
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(float)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = nn.Sequential(
        nn.Linear(X.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    torch.save(model.state_dict(), "model.pt")

def predict_neural(model_path: str, input_data: list[float]):
    state = torch.load(model_path, map_location="cpu")
    inp = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
    model = nn.Sequential(
        nn.Linear(len(input_data), 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        out = model(inp)
    return out.item()
