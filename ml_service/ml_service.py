import os
import io
import uuid
import base64
import json
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from trainers.utils import (
    ensure_numeric,
    train_test_split_xy,
    standardize_if_needed,
    metrics_report,
    save_artifact,
)
from trainers.nn_trainer import train_nn
from trainers.gp_trainer import train_gp
from trainers.rf_trainer import train_rf
from trainers.xgb_trainer import train_xgb
from trainers.transformer_trainer import train_transformer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UPLOAD_DIR = "/app/uploads"
ARTIFACT_DIR = "/app/artifacts"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

app = FastAPI(title="PetroAgent ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/models")
def models():
    return {
        "models": [
            "NeuralNet",
            "GaussianProcess",
            "RandomForest",
            "XGBoost",
            "Transformer",
        ]
    }

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    file_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
    data = await file.read()
    with open(path, "wb") as f:
        f.write(data)
    return {"file_id": file_id, "path": path, "rows": len(data)}

def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    df = pd.read_csv(path)
    return df

@app.post("/api/describe")
def describe(payload: Dict[str, Any]):
    path = payload.get("path") or os.path.join(UPLOAD_DIR, f"{payload['file_id']}.csv")
    df = _load_csv(path)
    preview = df.head(10).to_dict(orient="records")
    desc = df.describe(include="all").fillna("").to_dict()
    cols = list(df.columns)
    return {"columns": cols, "preview": preview, "describe": desc}

@app.post("/api/plots")
def plots(payload: Dict[str, Any]):
    path = payload.get("path") or os.path.join(UPLOAD_DIR, f"{payload['file_id']}.csv")
    features: List[str] = payload["features"]
    target: str = payload["target"]

    df = _load_csv(path)
    df = ensure_numeric(df, features + [target]).dropna()

    images: Dict[str, str] = {}

    # 1) Correlation heatmap (features + target)
    corr_cols = features + [target]
    corr = df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    im = ax.imshow(corr.values, cmap="viridis")
    ax.set_xticks(range(len(corr_cols))); ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right"); ax.set_yticklabels(corr_cols)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation heatmap")
    buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format="png"); plt.close(fig)
    images["corr"] = base64.b64encode(buf.getvalue()).decode()

    # 2) Feature histograms
    for col in features:
        fig, ax = plt.subplots(figsize=(5,3), dpi=140)
        ax.hist(df[col].values, bins=30)
        ax.set_title(f"Histogram: {col}")
        buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format="png"); plt.close(fig)
        images[f"hist_{col}"] = base64.b64encode(buf.getvalue()).decode()

    # 3) Target vs each feature scatter
    for col in features:
        fig, ax = plt.subplots(figsize=(5,3), dpi=140)
        ax.scatter(df[col].values, df[target].values, s=6, alpha=0.7)
        ax.set_xlabel(col); ax.set_ylabel(target)
        ax.set_title(f"{target} vs {col}")
        buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format="png"); plt.close(fig)
        images[f"scatter_{col}"] = base64.b64encode(buf.getvalue()).decode()

    return {"images": images}

@app.post("/api/train")
def train(payload: Dict[str, Any]):
    """
    payload = {
      "file_id": "...",
      "path": optional absolute path,
      "features": ["x1","x2"],
      "target": "y",
      "model": "NeuralNet" | "GaussianProcess" | "RandomForest" | "XGBoost" | "Transformer",
      "save_path": "/app/artifacts/my_model.pkl" (optional)
    }
    """
    path = payload.get("path") or os.path.join(UPLOAD_DIR, f"{payload['file_id']}.csv")
    features: List[str] = payload["features"]
    target: str = payload["target"]
    model_name: str = payload["model"]
    save_path: str = payload.get("save_path") or os.path.join(ARTIFACT_DIR, f"{uuid.uuid4()}_{model_name}.pkl")

    df = _load_csv(path)
    df = ensure_numeric(df, features + [target]).dropna()

    X_train, X_test, y_train, y_test = train_test_split_xy(df, features, target)
    X_train, X_test, scaler = standardize_if_needed(model_name, X_train, X_test)

    if model_name == "NeuralNet":
        model = train_nn(X_train, y_train, X_test, y_test)
    elif model_name == "GaussianProcess":
        model = train_gp(X_train, y_train)
    elif model_name == "RandomForest":
        model = train_rf(X_train, y_train)
    elif model_name == "XGBoost":
        model = train_xgb(X_train, y_train, X_test, y_test)
    elif model_name == "Transformer":
        model = train_transformer(X_train, y_train, X_test, y_test)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    # metrics
    metrics = metrics_report(model, X_train, y_train, X_test, y_test)

    # persist (with scaler if used)
    artefact_info = save_artifact(save_path, model, scaler)

    return {
        "model": model_name,
        "save_path": artefact_info["path"],
        "scaler_included": artefact_info["scaler_included"],
        "metrics": metrics,
    }
