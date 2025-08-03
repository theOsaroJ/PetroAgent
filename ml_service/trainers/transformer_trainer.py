import torch
import torch.nn as nn
import pandas as pd

model_trans = None

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_out = nn.Linear(32, 1)
    def forward(self, x):
        x = self.fc_in(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.fc_out(x)

def train_transformer(df, inputs, target):
    global model_trans
    X = torch.tensor(df[inputs].values, dtype=torch.float32)
    y = torch.tensor(df[target].values, dtype=torch.float32).unsqueeze(1)
    model = TransformerRegressor(len(inputs))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    for _ in range(50):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
    model_trans = model
    return model

def predict_transformer(df):
    X = torch.tensor(df.values, dtype=torch.float32)
    with torch.no_grad():
        return model_trans(X).numpy().flatten()
