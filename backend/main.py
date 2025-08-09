from fastapi import FastAPI, UploadFile, File
import pandas as pd

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    # return first 5 rows and headers
    return {
        "columns": df.columns.tolist(),
        "preview": df.head(5).to_dict(orient="records")
    }
