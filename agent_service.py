import os, uuid, base64, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import openai

# ML imports
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# --- Load OpenAI key ---
with open("api_key.txt") as f:
    openai.api_key = f.read().strip()

app = FastAPI(title="PetroAgent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# --- Schemas ---
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

class UploadResponse(BaseModel):
    id: str
    columns: List[str]

class TrainRequest(BaseModel):
    id: str
    features: List[str]
    target: str
    model_type: str
    save_path: str

class TrainResponse(BaseModel):
    mse: float
    plot_b64: str
    model_file: str

# --- Helpers ---
def save_plot(y_true, y_pred, uid):
    plt.figure(figsize=(6,4))
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn, mx = float(min(y_true)), float(max(y_true))
    plt.plot([mn, mx],[mn,mx], 'r--')
    plt.xlabel("True"); plt.ylabel("Pred")
    buf = BytesIO(); plt.tight_layout()
    plt.savefig(buf, format="png"); plt.close()
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

# --- Endpoints ---
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"user","content":req.message}]
    )
    return {"reply": resp.choices[0].message.content}

@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    path = f"data/{uid}.csv"
    contents = await file.read()
    with open(path, "wb") as f: f.write(contents)
    df = pd.read_csv(path)
    return {"id": uid, "columns": list(df.columns)}

@app.post("/api/train", response_model=TrainResponse)
def train(req: TrainRequest):
    path = f"data/{req.id}.csv"
    if not os.path.exists(path):
        raise HTTPException(404, "Data not found")
    df = pd.read_csv(path)
    X = df[req.features].values; y = df[req.target].values
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=42)

    # model selection
    if req.model_type=="neural_network":
        # deep MLP w/ BatchNorm + Dropout + early stop...
        class MLP(nn.Module):
            def __init__(self,in_dim):
                super().__init__()
                self.net = nn.Sequential(
                  nn.Linear(in_dim,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                  nn.Linear(128,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                  nn.Linear(64,32), nn.BatchNorm1d(32), nn.ReLU(),
                  nn.Linear(32,1)
                )
            def forward(self,x): return self.net(x)

        Xtr_t = torch.tensor(Xtr,dtype=torch.float32)
        ytr_t = torch.tensor(ytr,dtype=torch.float32).unsqueeze(1)
        ds = TensorDataset(Xtr_t,ytr_t)
        dl = DataLoader(ds,batch_size=32,shuffle=True)

        model = MLP(Xtr.shape[1])
        opt = torch.optim.Adam(model.parameters(),lr=1e-3)
        loss_fn = nn.MSELoss()
        best,pat=1e9,0
        for ep in range(200):
            model.train()
            for xb,yb in dl:
                opt.zero_grad(); loss_fn(model(xb),yb).backward(); opt.step()
            model.eval()
            val = float(loss_fn(model(Xtr_t),ytr_t))
            if val<best-1e-4:
                best,val,pat = val,val,0
                torch.save(model.state_dict(),f"models/{req.id}_nn.pt")
            else:
                pat+=1
            if pat>10: break
        model.load_state_dict(torch.load(f"models/{req.id}_nn.pt"))
        with torch.no_grad():
            preds = model(torch.tensor(Xte,dtype=torch.float32)).numpy().flatten()
        model_file = f"{req.save_path}/{req.id}_nn.pt"

    elif req.model_type=="random_forest":
        gs = GridSearchCV(RandomForestRegressor(random_state=0),
                          {"n_estimators":[100,300],"max_depth":[None,10,30]},
                          cv=3,n_jobs=-1)
        gs.fit(Xtr,ytr)
        model = gs.best_estimator_
        preds = model.predict(Xte)
        model_file = f"{req.save_path}/{req.id}_rf.joblib"
        joblib.dump(model, model_file)

    elif req.model_type=="gaussian_process":
        kern = RBF(1.0)+WhiteKernel(1.0)
        model = GaussianProcessRegressor(kernel=kern,normalize_y=True)
        model.fit(Xtr,ytr)
        preds = model.predict(Xte)
        model_file = f"{req.save_path}/{req.id}_gp.joblib"
        joblib.dump(model, model_file)

    elif req.model_type=="xgboost":
        xmod = xgb.XGBRegressor(objective="reg:squarederror",random_state=0)
        gs = GridSearchCV(xmod,
                          {"n_estimators":[100,300],"max_depth":[3,6],"learning_rate":[0.01,0.1]},
                          cv=3,n_jobs=-1)
        gs.fit(Xtr,ytr)
        model = gs.best_estimator_
        preds = model.predict(Xte)
        model_file = f"{req.save_path}/{req.id}_xgb.joblib"
        joblib.dump(model, model_file)

    else:
        raise HTTPException(400,"Unknown model_type")

    mse = mean_squared_error(yte:=yte, preds)
    plot_b64 = save_plot(yte, preds, req.id)
    return {"mse":mse, "plot_b64":plot_b64, "model_file":model_file}
