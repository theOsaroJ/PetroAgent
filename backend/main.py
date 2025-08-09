from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="PetroAgent Backend", version="1.0.0")

class Health(BaseModel):
    status: str
    service: str
    version: str

@app.get("/health", response_model=Health)
def health():
    return Health(status="ok", service="backend", version="1.0.0")

@app.get("/version")
def version():
    return {"version": "1.0.0"}
