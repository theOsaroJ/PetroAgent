# ml_service/ml_service.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
from pathlib import Path
import tempfile
import os

app = FastAPI(title="PetroAgent ML Service", version="1.0.0")

# CORS – dev-friendly defaults. Safe because the frontend is same-origin behind Nginx,
# but this also makes localhost testing work without preflight pain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten later for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ColumnsResponse(BaseModel):
    columns: List[str]

class TrainResponse(BaseModel):
    model_path: str
    model_type: str
    metrics: dict
    n_train: int
    n_test: int
    features: List[str]
    target: str

@app.get("/health")
def health():
    return {"status": "ok"}

def _to_tempfile(upload: UploadFile) -> str:
    # Persist UploadFile to a real temp file so pandas can read it reliably.
    suffix = Path(upload.filename or "data.csv").suffix or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = upload.file.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()
    # rewind the SpooledTemporaryFile to allow re-use in handlers if needed
    try:
        upload.file.seek(0)
    except Exception:
        pass
    return tmp.name

@app.post("/columns", response_model=ColumnsResponse)
async def detect_columns(file: UploadFile = File(...)):
    """
    Returns column names from the uploaded CSV (no data loaded).
    """
    tmp_path = _to_tempfile(file)
    try:
        # Only read header line to list columns fast
        df_head = pd.read_csv(tmp_path, nrows=0)
        cols = list(df_head.columns)
        if not cols:
            raise HTTPException(status_code=400, detail="No columns detected in file.")
        return {"columns": cols}
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass

def _decide_task(y: pd.Series) -> str:
    # Heuristic: numeric with many unique -> regression, else classification
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique(dropna=True) > 10:
            return "regression"
    return "classification"

def _make_model(task: str, name: str):
    name = (name or "").lower()
    if task == "regression":
        if name in ("rf", "randomforest", "random_forest"):
            return RandomForestRegressor(n_estimators=200, random_state=42)
        if name in ("mlp", "nn", "neural", "neuralnet", "neural_net"):
            return MLPRegressor(hidden_layer_sizes=(128, 64), random_state=42, max_iter=500)
        # default regressor
        return RandomForestRegressor(n_estimators=300, random_state=42)
    else:
        if name in ("rf", "randomforest", "random_forest"):
            return RandomForestClassifier(n_estimators=300, random_state=42)
        if name in ("logreg", "logistic", "logisticregression"):
            return LogisticRegression(max_iter=1000)
        if name in ("mlp", "nn", "neural", "neuralnet", "neural_net"):
            return MLPClassifier(hidden_layer_sizes=(128, 64), random_state=42, max_iter=500)
        # default classifier
        return RandomForestClassifier(n_estimators=300, random_state=42)

def _metrics(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if task == "regression":
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }
    else:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        }

@app.post("/train", response_model=TrainResponse)
async def train(
    file: UploadFile = File(...),
    features: str = Form(...),      # comma-separated
    target: str = Form(...),
    model: str = Form("random_forest"),
    save_dir: str = Form("./outputs"),
):
    """
    Train a model given a CSV file, selected features, and target.
    Returns metrics and where the model was saved (inside the container).
    """
    # Guard inputs
    features_list = [f.strip() for f in features.split(",") if f.strip()]
    if not features_list:
        raise HTTPException(status_code=400, detail="No features were provided.")
    if not target:
        raise HTTPException(status_code=400, detail="Target column is required.")

    tmp_path = _to_tempfile(file)
    try:
        df = pd.read_csv(tmp_path)
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    missing = [c for c in features_list + [target] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns in CSV: {missing}")

    X = df[features_list]
    y = df[target]

    task = _decide_task(y)
    model_obj = _make_model(task, model)

    # Basic split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if task == "classification" else None
    )

    # Fit
    model_obj.fit(X_train, y_train)

    # Predict & metrics
    y_pred = model_obj.predict(X_test)
    m = _metrics(task, y_test, y_pred)

    # Save model
    out_dir = Path(save_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(file.filename or "model").stem
    model_path = out_dir / f"{base}_{task}_{model_obj.__class__.__name__}.joblib"
    joblib.dump({"model": model_obj, "features": features_list, "target": target, "task": task}, model_path)

    # cleanup
    try:
        os.unlink(tmp_path)
    except FileNotFoundError:
        pass

    return {
        "model_path": str(model_path),
        "model_type": model_obj.__class__.__name__,
        "metrics": m,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": features_list,
        "target": target,
    }
