import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_neural(data_path: str, epochs: int = 10):
    df = pd.read_csv(data_path)
    X  = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y  = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)
    dl = DataLoader(TensorDataset(X,y), batch_size=32, shuffle=True)

    model   = nn.Sequential(nn.Linear(X.shape[1],64), nn.ReLU(), nn.Linear(64,1))
    opt     = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        for xb,yb in dl:
            pred  = model(xb)
            loss  = loss_fn(pred,yb)
            opt.zero_grad(); loss.backward(); opt.step()

    torch.save(model.state_dict(), "neural.pt")

def predict_neural(model_path: str, input_data: list[float]):
    state = torch.load(model_path, map_location="cpu")
    model = nn.Sequential(nn.Linear(len(input_data),64), nn.ReLU(), nn.Linear(64,1))
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        return model(torch.tensor(input_data,dtype=torch.float32)).item()
