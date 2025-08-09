import os
import uuid
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

DATA_DIR = os.environ.get("DATA_DIR", "/data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

app = FastAPI(title="PetroAgent Backend")

# If you access this service directly (not via nginx), CORS it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

# Serve /files/* from /data
app.mount("/files", StaticFiles(directory=DATA_DIR), name="files")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are supported.")
    file_id = str(uuid.uuid4()) + ".csv"
    dest_path = os.path.join(UPLOADS_DIR, file_id)
    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)
    try:
        df = pd.read_csv(dest_path, nrows=50)
    except Exception as e:
        raise HTTPException(400, f"CSV parse failed: {e}")

    head = df.head(10).to_dict(orient="records")
    columns = list(df.columns)

    # Return path relative to /data for cross-service access
    rel_path = os.path.relpath(dest_path, DATA_DIR).replace("\\", "/")
    return {"file_rel_path": rel_path, "columns": columns, "head": head}
