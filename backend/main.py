from fastapi import FastAPI, UploadFile, File
import pandas as pd

app = FastAPI()

@app.get('/')
def health():
    return {'status': 'ok'}

@app.post('/upload')
async def upload_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, nrows=5)  # peek first 5 rows
    return {
        'filename': file.filename,
        'headers': list(df.columns),
        'preview': df.head(3).to_dict(orient='records')
    }
