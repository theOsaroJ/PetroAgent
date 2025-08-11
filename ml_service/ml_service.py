from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
import io
import joblib

# Models
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

app = FastAPI(title="PetroAgent ML Service")

# --- CORS: allow UI and nginx to preflight (OPTIONS) and POST without 405 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # lock this down if you need to
    allow_credentials=True,
    allow_methods=["*"],          # <- ensures OPTIONS is allowed too
    allow_headers=["*"],
)

# Utility: build a model from a short name
def build_model(name: str):
    name = (name or "").lower().strip()
    if name in ["rf", "randomforest", "random_forest", "random forest"]:
        return RandomForestRegressor(n_estimators=200, random_state=42)
    if name in ["linreg", "linear", "linear_regression", "linear regression"]:
        return LinearRegression()
    # default
    return RandomForestRegressor(n_estimators=200, random_state=42)

# Detect columns from uploaded CSV
@app.post("/columns")
async def detect_columns(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
        cols = list(df.columns)
        return {"columns": cols}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Could not parse CSV: {e}"})

# Train endpoint
@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    features: str = Form(...),       # comma-separated list: "f1,f2,f3"
    target: str = Form(...),
    model_name: str = Form("RandomForest"),
    save_dir: str = Form("/app/outputs"),  # path inside the ml_service container
):
    try:
        # read data
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))

        feats = [f.strip() for f in features.split(",") if f.strip()]
        if not feats:
            return JSONResponse(status_code=400, content={"error": "No features provided"})
        if target not in df.columns:
            return JSONResponse(status_code=400, content={"error": f"Target '{target}' not found"})

        X = df[feats]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = build_model(model_name)
        model.fit(X_train, y_train)

        # simple metrics
        preds = model.predict(X_test)
        r2 = float(r2_score(y_test, preds))
        rmse = float(mean_squared_error(y_test, preds, squared=False))

        # save
        os.makedirs(save_dir, exist_ok=True)
        safe_name = model_name.lower().replace(" ", "_")
        model_path = os.path.join(save_dir, f"{safe_name}.joblib")
        joblib.dump(
            {
                "model": model,
                "features": feats,
                "target": target,
                "metrics": {"r2": r2, "rmse": rmse},
            },
            model_path,
        )

        return {
            "ok": True,
            "model_path": model_path,
            "metrics": {"r2": r2, "rmse": rmse},
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
