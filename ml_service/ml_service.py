from fastapi import FastAPI
from trainers.xgboost_trainer import train_xgboost, predict_xgboost
from trainers.neural_trainer import train_neural, predict_neural

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ml ok"}

@app.post("/train/xgb")
def train_xgb():
    result = train_xgboost()
    return result

@app.post("/predict/xgb")
def predict_xgb():
    result = predict_xgboost()
    return result

@app.post("/train/neural")
def train_neural_net():
    result = train_neural()
    return result

@app.post("/predict/neural")
def predict_neural_net():
    result = predict_neural()
    return result
