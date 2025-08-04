import os, io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd, requests

app = FastAPI()
BACKEND = "http://backend:5000"
ML = "http://ml_service:8000"

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    with open("latest_upload.csv","wb") as f: f.write(data)
    cols = pd.read_csv(io.StringIO(data.decode())).columns.tolist()
    return {"columns": cols}

@app.post("/api/chat")
async def chat(prompt: str = Form(...)):
    r = requests.post(f"{BACKEND}/api/chat", json={"prompt": prompt})
    return JSONResponse(r.json())

@app.post("/api/train")
async def train(
    model_type: str = Form(...),
    features: str = Form(...),
    target: str = Form(...),
    save_path: str = Form(...)
):
    feats = features.split(",")
    payload = {
      "model_type": model_type,
      "features": feats,
      "target": target,
      "save_path": save_path
    }
    r = requests.post(f"{ML}/api/train", json=payload)
    return JSONResponse(r.json())

@app.get("/api/plot")
def plot(feature: str):
    r = requests.get(f"{ML}/api/plot", params={"feature": feature})
    return StreamingResponse(io.BytesIO(r.content), media_type="image/png")
