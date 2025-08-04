import io, os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import torch, torch.nn as nn
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode()))
    # Save for later
    with open("latest_upload.csv","wb") as f: f.write(content)
    return {"columns": df.columns.tolist()}

class TrainRequest(BaseModel):
    model_type: str
    features: list[str]
    target: str
    save_path: str

@app.post("/api/train")
def train(req: TrainRequest):
    df = pd.read_csv("latest_upload.csv")
    X, y = df[req.features], df[req.target]
    if req.model_type == "random_forest":
        m = RandomForestRegressor(n_estimators=200)
        m.fit(X, y)
        joblib.dump(m, req.save_path)
    elif req.model_type == "xgboost":
        m = XGBRegressor(n_estimators=200, tree_method="hist")
        m.fit(X, y)
        joblib.dump(m, req.save_path)
    elif req.model_type == "neural_net":
        class Net(nn.Module):
            def __init__(self, D):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(D,128), nn.ReLU(),
                    nn.Linear(128,64), nn.ReLU(),
                    nn.Linear(64,1)
                )
            def forward(self, x): return self.layers(x)
        net = Net(X.shape[1])
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        data = torch.tensor(X.values, dtype=torch.float32)
        target = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
        for epoch in range(100):
            optimizer.zero_grad()
            out = net(data)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
        torch.save(net.state_dict(), req.save_path)
    elif req.model_type == "transformer":
        # placeholder: fine-tune for text target
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
        # (actual fine-tuning skipped for brevity)
        model.save_pretrained(req.save_path)
    else:
        return {"error":"unknown model"}
    return {"status":"trained", "model": req.model_type}

@app.get("/api/plot")
def plot(feature: str):
    df = pd.read_csv("latest_upload.csv")
    plt.figure()
    df[feature].hist()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
