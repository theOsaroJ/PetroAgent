import os
from fastapi import FastAPI
from pydantic import BaseModel
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    resp = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role":"user","content":request.prompt}]
    )
    return {"reply": resp.choices[0].message.content}
