from __future__ import annotations
import os, joblib
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Any, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def train_test_split_xy(df: pd.DataFrame, features: List[str], target: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def standardize_if_needed(model_name: str, X_train: np.ndarray, X_test: np.ndarray):
    scaler = None
    if model_name in ("NeuralNet", "GaussianProcess", "Transformer", "XGBoost"):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, scaler

def metrics_report(model, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    def _pred(m, X):
        try:
            return m.predict(X)
        except Exception:
            # PyTorch models wrapped
            return m(X)

    yhat_tr = _pred(model, X_train)
    yhat_te = _pred(model, X_test)

    mae = float(mean_absolute_error(y_test, yhat_te))
    rmse = float(mean_squared_error(y_test, yhat_te, squared=False))
    r2 = float(r2_score(y_test, yhat_te))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def save_artifact(path: str, model: Any, scaler: Optional[Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model": model, "scaler": scaler}
    joblib.dump(payload, path)
    return {"path": path, "scaler_included": scaler is not None}
