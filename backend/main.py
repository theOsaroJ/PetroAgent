from fastapi import FastAPI, UploadFile, File
import pandas as pd

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    # Read first 100 rows to detect headers & preview
    df = pd.read_csv(file.file, nrows=100)
    return {
        "filename": file.filename,
        "headers": list(df.columns),
        "preview": df.head(5).to_dict(orient="records")
    }
