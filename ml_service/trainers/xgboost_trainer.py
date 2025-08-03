import os, joblib
from xgboost import XGBRegressor
import pandas as pd

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_xgboost(X: pd.DataFrame, y: pd.Series):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    path = f"{MODEL_DIR}/xgb.pkl"
    joblib.dump(model, path)
    return {"model_path": path, "train_score": model.score(X,y)}

def predict_xgboost(df: pd.DataFrame):
    model = joblib.load(f"{MODEL_DIR}/xgb.pkl")
    preds = model.predict(df)
    return {"predictions": preds.tolist()}
