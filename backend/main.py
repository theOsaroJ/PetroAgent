from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class HelloRequest(BaseModel):
    name: str

@app.get("/")
def root():
    return {"message": "Backend up"}

@app.post("/hello")
def hello(req: HelloRequest):
    return {"message": f"Hello, {req.name}!"}
