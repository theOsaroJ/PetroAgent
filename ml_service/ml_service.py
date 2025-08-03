from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os

from trainers.neural_trainer     import train_neural,     predict_neural
from trainers.rf_trainer         import train_rf,         predict_rf
from trainers.gp_trainer         import train_gp,         predict_gp
from trainers.transformer_trainer import train_transformer, predict_transformer

class TrainRequest(BaseModel):
    data: list
    input_columns: list[str]
    target_column: str

class PredictRequest(BaseModel):
    data: list
    input_columns: list[str]

app = FastAPI()

@app.post("/train/{model_type}")
def train(model_type: str, req: TrainRequest):
    df = pd.DataFrame(req.data, columns=req.input_columns + [req.target_column])
    if model_type == "neural":
        train_neural(df, req.input_columns, req.target_column)
    elif model_type == "random_forest":
        train_rf(df, req.input_columns, req.target_column)
    elif model_type == "gp":
        train_gp(df, req.input_columns, req.target_column)
    elif model_type == "transformer":
        train_transformer(df, req.input_columns, req.target_column)
    else:
        return {"error": "Unknown model_type"}
    return {"status": "trained", "model_type": model_type}

@app.post("/predict/{model_type}")
def predict(model_type: str, req: PredictRequest):
    df = pd.DataFrame(req.data, columns=req.input_columns)
    if model_type == "neural":
        preds = predict_neural(df)
    elif model_type == "random_forest":
        preds = predict_rf(df)
    elif model_type == "gp":
        preds = predict_gp(df)
    elif model_type == "transformer":
        preds = predict_transformer(df)
    else:
        return {"error": "Unknown model_type"}
    return {"predictions": preds.tolist()}
