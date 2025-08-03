from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd

from trainers.neural_trainer       import train_neural
from trainers.tree_trainer         import train_tree
from trainers.gp_trainer           import train_gp
from trainers.transformer_trainer  import train_transformer

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

    elif model_type in {"random_forest", "xgboost", "linear"}:
        model, score = train_tree(model_type, X, y)
        return {"model": model_type, "score": score}

    elif model_type == "gp":
        model, score = train_gp(X, y)
        return {"model": "gp", "score": score}

    elif model_type == "transformer":
        history = train_transformer(X, y)
        return {"model": "transformer", "history": history}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown model_type: {model_type}")
