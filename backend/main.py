from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any

app = FastAPI()

class DataRequest(BaseModel):
    data: List[List[Any]]
    input_columns: List[str]
    target_column: str

@app.post("/api/preprocess")
def preprocess(req: DataRequest):
    return {
        "columns": req.input_columns,
        "n_samples": len(req.data)
    }

@app.get("/api/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
