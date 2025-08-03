import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

model_neural = None

class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_neural(df: pd.DataFrame, inputs, target):
    global model_neural
    X = torch.tensor(df[inputs].values, dtype=torch.float32)
    y = torch.tensor(df[target].values, dtype=torch.float32).unsqueeze(1)
    model = Net(len(inputs))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(100):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
    model_neural = model
    return model

def predict_neural(df):
    X = torch.tensor(df.values, dtype=torch.float32)
    with torch.no_grad():
        return model_neural(X).numpy().flatten()
