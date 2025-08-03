import os, joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_classification(X: pd.DataFrame, y: pd.Series):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    path = f"{MODEL_DIR}/clf.pkl"
    joblib.dump(model, path)
    return {"model_path": path, "train_score": model.score(X,y)}

def predict_classification(df: pd.DataFrame):
    model = joblib.load(f"{MODEL_DIR}/clf.pkl")
    preds = model.predict(df)
    return {"predictions": preds.tolist()}
