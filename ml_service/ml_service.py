from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd, io

from trainers.neural_trainer import train_neural, predict_neural
from trainers.rf_trainer     import train_rf,     predict_rf
from trainers.linear_trainer import train_linear, predict_linear
from trainers.xgb_trainer    import train_xgboost, predict_xgboost

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/train")
async def train(
    file: UploadFile = File(...),
    algorithm: str      = Form(...),
    epochs: int         = Form(10)
):
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode()))
        df.to_csv("uploaded.csv", index=False)
        alg = algorithm.lower()
        if   alg == "neural":   train_neural("uploaded.csv", epochs)
        elif alg == "rf":        train_rf("uploaded.csv")
        elif alg == "linear":    train_linear("uploaded.csv")
        elif alg == "xgboost":   train_xgboost("uploaded.csv")
        else: raise HTTPException(400, f"Unknown algorithm {algorithm}")
        return {"status": "trained", "algorithm": alg}
    except Exception as e:
        raise HTTPException(400, detail=str(e))

@app.post("/predict")
async def predict(req: dict):
    try:
        alg        = req.get("algorithm","").lower()
        model_path = req.get("model_path","")
        inp        = req.get("input_data",[])
        if   alg == "neural":   p = predict_neural(model_path, inp)
        elif alg == "rf":        p = predict_rf(model_path, inp)
        elif alg == "linear":    p = predict_linear(model_path, inp)
        elif alg == "xgboost":   p = predict_xgboost(model_path, inp)
        else: raise HTTPException(400, f"Unknown algorithm {alg}")
        return {"prediction": p}
    except Exception as e:
        raise HTTPException(400, detail=str(e))
