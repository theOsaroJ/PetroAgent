from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="PetroAgent Backend")

class Ping(BaseModel):
    msg: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ping")
def ping(p: Ping):
    return {"echo": p.msg}
