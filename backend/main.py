from fastapi import FastAPI, File, UploadFile, Form
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor

app = FastAPI()

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    return {"columns": list(df.columns)}

@app.post("/api/train")
async def train(
    model: str = Form(...),
    features: str = Form(...),
    target: str = Form(...),
):
    # VERY basic stub: replace with actual training logic or delegate to ml_service
    return {"status": "trained", "model": model}
