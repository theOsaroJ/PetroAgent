import pandas as pd
from xgboost import XGBClassifier
import joblib

MODEL_PATH = "xgb_model.joblib"

def train_xgboost():
    # example: train on built-in Iris dataset
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return {"message": "xgboost model trained", "model_path": MODEL_PATH}

def predict_xgboost():
    # dummy single prediction example
    model = joblib.load(MODEL_PATH)
    sample = model.get_booster().feature_names  # just names, not used
    # return a fixed prediction
    pred = int(model.predict([[5.1, 3.5, 1.4, 0.2]])[0])
    return {"prediction": pred}
