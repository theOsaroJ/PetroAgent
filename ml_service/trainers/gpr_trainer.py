import os, joblib
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_gpr(X: pd.DataFrame, y: pd.Series):
    model = GaussianProcessRegressor()
    model.fit(X, y)
    path = f"{MODEL_DIR}/gpr.pkl"
    joblib.dump(model, path)
    return {"model_path": path, "train_score": model.score(X,y)}

def predict_gpr(df: pd.DataFrame):
    model = joblib.load(f"{MODEL_DIR}/gpr.pkl")
    preds, _ = model.predict(df, return_std=True)
    return {"predictions": preds.tolist()}
