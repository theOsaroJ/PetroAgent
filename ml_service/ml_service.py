from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import joblib, os, json

app = FastAPI(title="PetroAgent - ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_csv(upload: UploadFile) -> pd.DataFrame:
    try:
        data = upload.file.read()
        if not data:
            raise ValueError("Empty file")
        df = pd.read_csv(BytesIO(data))
        if df.empty:
            raise ValueError("CSV has no rows")
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read CSV: {e}")
    finally:
        try:
            upload.file.close()
        except Exception:
            pass

@app.post("/ml/columns")
def get_columns(file: UploadFile = File(...)):
    df = read_csv(file)
    return {"columns": list(map(str, df.columns.tolist()))}

def build_model(model_type: str):
    mt = (model_type or '').lower()
    if mt == "random_forest":
        return RandomForestRegressor(n_estimators=300, random_state=42)
    if mt == "xgboost":
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            random_state=42,
        )
    if mt == "gaussian_process":
        kernel = C(1.0, (1e-2, 1e3)) * RBF(length_scale=1.0)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
    if mt == "mlp" or mt == "neural_net":
        return MLPRegressor(hidden_layer_sizes=(256,128,64), activation='relu',
                            alpha=1e-4, learning_rate_init=1e-3, max_iter=500,
                            early_stopping=True, random_state=42)
    if mt == "transformer":
        # Fallback if torch isn't available: use MLP
        return MLPRegressor(hidden_layer_sizes=(384,192,96), activation='relu',
                            alpha=1e-4, learning_rate_init=8e-4, max_iter=600,
                            early_stopping=True, random_state=42)
    raise HTTPException(status_code=400, detail=f"Unknown model_type '{model_type}'")

@app.post("/ml/train")
def train(
    file: UploadFile = File(...),
    features: str = Form(..., description="JSON list of feature column names"),
    target: str = Form(..., description="Target column name"),
    model_type: str = Form("random_forest"),
    save_dir: str = Form("/app/outputs")
):
    df = read_csv(file)

    try:
        feat_list: List[str] = json.loads(features)
        if not isinstance(feat_list, list):
            raise ValueError
        feat_list = [str(c) for c in feat_list]
    except Exception:
        raise HTTPException(status_code=400, detail="`features` must be a JSON array of strings")

    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target '{target}' not in CSV columns")

    # Drop rows with NA in used columns
    used = feat_list + [target]
    df = df[used].dropna()

    X = df[feat_list].select_dtypes(include=[np.number])
    if X.shape[1] != len(feat_list):
        raise HTTPException(status_code=400, detail="All selected features must be numeric")

    y = df[target]
    if not np.issubdtype(y.dtype, np.number):
        # try to coerce
        try:
            y = pd.to_numeric(y)
        except Exception:
            raise HTTPException(status_code=400, detail="Target must be numeric")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(model_type)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    # Save
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(file.filename or "dataset")[0]
    out_path = os.path.join(save_dir, f"{base}_{model_type}.joblib")
    payload = {
        "model_type": model_type,
        "feature_names": feat_list,
        "target": target
    }
    joblib.dump({"sk_model": model, "meta": payload}, out_path)

    return {
        "model_type": model_type,
        "saved_to": out_path,
        "n_samples": int(len(df)),
        "metrics": {"rmse": rmse, "r2": r2}
    }
