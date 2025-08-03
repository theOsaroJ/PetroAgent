from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Columns(BaseModel):
    columns: list[str]

@app.post("/api/upload", response_model=Columns)
async def upload_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    return Columns(columns=df.columns.tolist())
