import os
import io
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from utils import ensure_dir, split_xy, save_metrics_json, parity_plot
from trainers import rf, xgb, gp, nn, transformer

app = FastAPI(title="PetroAgent ML Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

DEFAULT_SAVE_DIR = "/app/outputs"

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/columns")
async def columns(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
        cols = df.columns.tolist()
        sample = df.head(10).to_dict(orient="records")
        return {"columns": cols, "sample": sample, "rows": int(df.shape[0])}
    except Exception as e:
        raise HTTPException(400, f"Failed to parse CSV: {e}")

@app.post("/train")
async def train(
    file: UploadFile = File(...),
    features: str = Form(...),
    target: str = Form(...),
    model_type: str = Form(...),
    save_dir: Optional[str] = Form(None),
):
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Failed to read CSV: {e}")

    feat_list = [c.strip() for c in features.split(",") if c.strip()]
    if target not in df.columns:
        raise HTTPException(400, f"Target '{target}' not in CSV.")
    for f in feat_list:
        if f not in df.columns:
            raise HTTPException(400, f"Feature '{f}' not in CSV.")

    X, y = split_xy(df, feat_list, target)
    out_dir = ensure_dir(save_dir or DEFAULT_SAVE_DIR)
    model_type = model_type.lower().strip()

    # Train
    if model_type == "rf":
        result = rf.train_rf(X, y, out_dir)
    elif model_type == "xgb":
        result = xgb.train_xgb(X, y, out_dir)
    elif model_type == "gp":
        result = gp.train_gp(X, y, out_dir)
    elif model_type == "nn":
        result = nn.train_mlp(X, y, out_dir)
    elif model_type == "transformer":
        result = transformer.train_transformer(X, y, out_dir)
    else:
        raise HTTPException(400, f"Unknown model_type: {model_type}")

    # Parity plot
    parity_path = os.path.join(out_dir, f"{model_type}_parity.png")
    parity_plot(result["y_true"], result["y_pred"], parity_path)

    # Persist metrics json
    metrics_path = os.path.join(out_dir, f"{model_type}_metrics.json")
    save_metrics_json(result["metrics"], metrics_path)

    payload = {
        "model_type": model_type,
        "metrics": result["metrics"],
        "artifacts": {
            "model_path": result["model_path"],
            "parity_plot": parity_path,
            "metrics_json": metrics_path,
        }
    }
    return payload
