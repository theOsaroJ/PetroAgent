import torch, torch.nn as nn, pandas as pd
import joblib
from sklearn.model_selection import train_test_split

class MLP(nn.Module):
    def __init__(self, input_dim, layers, activations):
        super().__init__()
        seq = []
        dim = input_dim
        for l, act in zip(layers, activations):
            seq.append(nn.Linear(dim, l))
            seq.append(getattr(nn, act)())
            dim = l
        seq.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)

def train_nn(features, target, params):
    df = pd.read_csv("data.csv")
    X = df[features].values; y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = MLP(len(features), params["layers"], params["activations"])
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()
    for epoch in range(params["epochs"]):
        pred = model(torch.tensor(X_train, dtype=torch.float32))
        loss = loss_fn(pred.squeeze(), torch.tensor(y_train, dtype=torch.float32))
        opt.zero_grad(); loss.backward(); opt.step()
    joblib.dump(model, params["save_path"])
    return {"status": "trained", "mse": loss.item()}
