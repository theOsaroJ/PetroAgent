from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
def chat(req: PromptRequest):
    res = requests.post(
        'http://agent:7000/chat',
        json={'prompt': req.prompt}
    )
    return res.json()
