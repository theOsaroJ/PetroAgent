import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI

def load_openai_key() -> str:
    # 1) secret file
    f = os.environ.get("OPENAI_API_KEY_FILE", "/run/secrets/openai_key")
    if f and os.path.exists(f):
        return open(f, "r").read().strip()
    # 2) local file
    if os.path.exists("api_key.txt"):
        return open("api_key.txt", "r").read().strip()
    # 3) env var string
    k = os.environ.get("OPENAI_API_KEY", "").strip()
    if k:
        return k
    raise RuntimeError("OpenAI API key missing. Provide api_key.txt or env OPENAI_API_KEY or mount /run/secrets/openai_key")

OPENAI_KEY = load_openai_key()
MODEL = os.environ.get("OPENAI_API_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_KEY)

app = FastAPI(title="PetroAgent Chat")

class ChatPayload(BaseModel):
    message: str
    context: Optional[str] = ""

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL}

@app.post("/chat")
def chat(p: ChatPayload):
    try:
        sys = (
            "You are PetroAgent, a petroleum engineering assistant that can discuss drilling, production, "
            "reservoir engineering, CSV column meaning, and help choose ML models (neural nets, transformers, "
            "random forests, XGBoost, Gaussian processes). Keep answers crisp and actionable."
        )
        ctx = p.context or ""
        msgs = [
            {"role": "system", "content": sys + (f" Dataset columns and context: {ctx}" if ctx else "")},
            {"role": "user", "content": p.message}
        ]
        resp = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0.2)
        text = resp.choices[0].message.content
        return {"reply": text}
    except Exception as e:
        raise HTTPException(500, f"Chat failed: {e}")
