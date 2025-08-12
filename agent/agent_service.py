from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

from openai import OpenAI

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

app = FastAPI(title="PetroAgent - Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    message: str

@app.post("/api/chat")
def chat_endpoint(payload: ChatIn):
    if not OPENAI_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    try:
        client = OpenAI(api_key=OPENAI_KEY)
        msg = payload.message.strip()
        if not msg:
            return {"reply": "Please type a message."}
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are PetroAgent, a petroleum engineering assistant and you know everything about petroluem engineering"},
                {"role": "user", "content": msg}
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        return {"reply": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
