from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from trainers import (
    neural_net,
    random_forest,
    gp,
    xgboost_trainer,
    transformer,
)

app = FastAPI(title="PetroAgent ML Service")

class TrainRequest(BaseModel):
    features: list[str]
    target: str
    model_type: str  # "nn","rf","gp","xgb","tf"

@app.post("/train")
async def train(req: TrainRequest, file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    X = df[req.features]
    y = df[req.target]
    if req.model_type == "nn":
        model = neural_net.train(X, y)
    elif req.model_type == "rf":
        model = random_forest.train(X, y)
    elif req.model_type == "gp":
        model = gp.train(X, y)
    elif req.model_type == "xgb":
        model = xgboost_trainer.train(X, y)
    elif req.model_type == "tf":
        model = transformer.train(X, y)
    else:
        raise ValueError("Unknown model type")
    path = f"./models/{req.model_type}.pt"
    model.save(path)
    return {"status": "trained", "model_path": path}
