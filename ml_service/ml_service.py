import os
import json
import uuid
import pandas as pd
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from trainers import (
    train_neural_net,
    train_random_forest,
    train_xgboost,
    train_gp,
    train_tab_transformer,
)
from trainers.utils import ensure_subdir, sanitize_rel, save_plot_paths

DATA_DIR = os.environ.get("DATA_DIR", "/data")

app = FastAPI(title="PetroAgent ML Service")

class TrainPayload(BaseModel):
    file_rel_path: str
    features: List[str]
    target: str
    model_type: str = Field(pattern="^(neural_net|random_forest|xgboost|gp|tab_transformer)$")
    save_dir: str = "artifacts/run1"
    params: Dict[str, Any] = {}
    test_size: float = 0.2
    random_state: int = 42

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/train")
def train(payload: TrainPayload):
    # resolve absolute path to CSV
    csv_path = os.path.join(DATA_DIR, payload.file_rel_path)
    if not os.path.exists(csv_path):
        raise HTTPException(400, f"CSV not found: {payload.file_rel_path}")

    # sanitize and create save directories
    rel_art_dir = sanitize_rel(payload.save_dir)
    abs_art_dir = os.path.join(DATA_DIR, rel_art_dir)
    ensure_subdir(abs_art_dir)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(400, f"Failed to read CSV: {e}")

    features = payload.features
    target = payload.target
    for c in features + [target]:
        if c not in df.columns:
            raise HTTPException(400, f"Column missing in CSV: {c}")

    X = df[features]
    y = df[target]

    model_type = payload.model_type

    if model_type == "neural_net":
        out = train_neural_net(X, y, abs_art_dir, test_size=payload.test_size, random_state=payload.random_state, **payload.params)
    elif model_type == "random_forest":
        out = train_random_forest(X, y, abs_art_dir, test_size=payload.test_size, random_state=payload.random_state, **payload.params)
    elif model_type == "xgboost":
        out = train_xgboost(X, y, abs_art_dir, test_size=payload.test_size, random_state=payload.random_state, **payload.params)
    elif model_type == "gp":
        out = train_gp(X, y, abs_art_dir, test_size=payload.test_size, random_state=payload.random_state, **payload.params)
    elif model_type == "tab_transformer":
        out = train_tab_transformer(X, y, abs_art_dir, test_size=payload.test_size, random_state=payload.random_state, **payload.params)
    else:
        raise HTTPException(400, f"Unknown model_type {model_type}")

    # convert absolute artifact paths to /data-relative for download via /backend/files/*
    rel_artifacts = save_plot_paths(out["artifacts"], DATA_DIR)

    # write metadata json
    run_id = str(uuid.uuid4())[:8]
    meta_path = os.path.join(abs_art_dir, f"run_meta_{run_id}.json")
    with open(meta_path, "w") as f:
        json.dump({
            "model_type": model_type,
            "features": features,
            "target": target,
            "metrics": out["metrics"],
            "artifacts": rel_artifacts,
            "model_path": os.path.relpath(out["model_path"], DATA_DIR).replace("\\", "/")
        }, f, indent=2)

    rel_meta = os.path.relpath(meta_path, DATA_DIR).replace("\\", "/")
    return {
        "metrics": out["metrics"],
        "artifacts": rel_artifacts + [rel_meta],
        "model_path": os.path.relpath(out["model_path"], DATA_DIR).replace("\\", "/"),
        "notes": out.get("notes", "")
    }
