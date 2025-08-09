from fastapi import FastAPI
from pydantic import BaseModel
from trainers.neural_net import train_nn
from trainers.gaussian_process import train_gp
from trainers.random_forest import train_rf
from trainers.xgboost_trainer import train_xgb
from trainers.transformer import train_transformer

class TrainConfig(BaseModel):
    features: list[str]
    target: str
    model_type: str
    params: dict

app = FastAPI()

@app.post("/train")
def train(cfg: TrainConfig):
    if cfg.model_type == "NeuralNet":
        return train_nn(cfg.features, cfg.target, cfg.params)
    if cfg.model_type == "GP":
        return train_gp(cfg.features, cfg.target, cfg.params)
    if cfg.model_type == "RF":
        return train_rf(cfg.features, cfg.target, cfg.params)
    if cfg.model_type == "XGBoost":
        return train_xgb(cfg.features, cfg.target, cfg.params)
    if cfg.model_type == "Transformer":
        return train_transformer(cfg.features, cfg.target, cfg.params)
    return {"error": "Unknown model"}
