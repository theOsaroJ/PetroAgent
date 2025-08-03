from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from trainers.neural_trainer import train_neural, predict_neural

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    data_path: str
    epochs: int

class PredictRequest(BaseModel):
    model_path: str
    input_data: list[float]

@app.post("/train")
def train(req: TrainRequest):
    try:
        train_neural(req.data_path, epochs=req.epochs)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = predict_neural(req.model_path, req.input_data)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
