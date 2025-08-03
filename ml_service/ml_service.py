from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd, io

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

app = FastAPI()

class MLRequest(BaseModel):
    model: str
    csv_data: str
    input_cols: List[str]
    target_col: str

class TabTransformerRegressor(nn.Module):
    def __init__(self, n_features, dim_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(1, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(dim_model * n_features, 1)

    def forward(self, x):
        b, f = x.shape
        x = x.unsqueeze(2)
        x = self.embed(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2).reshape(b, -1)
        return self.head(x).squeeze(1)

@app.post("/train_predict")
def train_predict(req: MLRequest):
    df = pd.read_csv(io.StringIO(req.csv_data))
    X = df[req.input_cols].values
    y = df[req.target_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    if req.model == "neural_network":
        m = MLPRegressor(max_iter=500)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

    elif req.model == "random_forest":
        m = RandomForestRegressor()
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

    elif req.model == "gp":
        m = GaussianProcessRegressor()
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

    elif req.model == "xgboost":
        m = XGBRegressor(use_label_encoder=False, eval_metric="rmse")
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

    elif req.model == "transformer":
        tr = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        loader = DataLoader(tr, batch_size=32, shuffle=True)
        model = TabTransformerRegressor(n_features=X_train.shape[1])
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(20):
            for xb, yb in loader:
                optim.zero_grad()
                loss_fn(model(xb), yb).backward()
                optim.step()

        model.eval()
        with torch.no_grad():
            tst = torch.tensor(X_test, dtype=torch.float32)
            preds = model(tst).numpy()

    else:
        return {"error": f"Unknown model '{req.model}'."}

    return {"model": req.model, "r2_score": float(r2_score(y_test, preds))}
