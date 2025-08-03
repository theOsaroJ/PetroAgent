from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os

from trainers.xgboost_trainer import train_xgboost, predict_xgboost
from trainers.gpr_trainer      import train_gpr, predict_gpr
from trainers.classification_trainer import train_classification, predict_classification
from trainers.neural_trainer   import train_neural, predict_neural

app = FastAPI()

class TrainReq(BaseModel):
    file_path: str
    target_column: str
    feature_columns: list[str]

class PredictReq(BaseModel):
    data: list[dict]

@app.post("/train/{model_type}")
async def train(model_type: str, req: TrainReq):
    if not os.path.exists(req.file_path):
        raise HTTPException(400, "File not found")
    df = pd.read_csv(req.file_path)
    X, y = df[req.feature_columns], df[req.target_column]
    if model_type == "xgboost":      return train_xgboost(X,y)
    if model_type == "gpr":          return train_gpr(X,y)
    if model_type == "classification": return train_classification(X,y)
    if model_type == "neural":       return train_neural(X,y)
    raise HTTPException(400, "Unknown model type")

@app.post("/predict/{model_type}")
async def predict(model_type: str, req: PredictReq):
    df = pd.DataFrame(req.data)
    if model_type == "xgboost":      return predict_xgboost(df)
    if model_type == "gpr":          return predict_gpr(df)
    if model_type == "classification": return predict_classification(df)
    if model_type == "neural":       return predict_neural(df)
    raise HTTPException(400, "Unknown model type")
