from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
from trainers.neural_trainer import train_neural, predict_neural
from trainers.tree_trainer import train_tree, predict_tree

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    return {"columns": df.columns.tolist()}

@app.post("/train")
async def train(model_type: str, features: list[str], target: str, file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    X, y = df[features], df[target]
    if model_type == "neural":
        history = train_neural(X, y)
        return {"model": "neural", "history": history}
    else:
        model, score = train_tree(model_type, X, y)
        return {"model": model_type, "score": score}
