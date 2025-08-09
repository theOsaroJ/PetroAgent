import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

def read_key_from_file() -> str:
    key_file = os.getenv("OPENAI_KEY_FILE", "")
    if not key_file or not os.path.isfile(key_file):
        raise RuntimeError("OPENAI_KEY_FILE not set or file not found")
    with open(key_file, "r") as f:
        k = f.read().strip()
        if not k:
            raise RuntimeError("OpenAI API key file is empty")
        return k

OPENAI_KEY = read_key_from_file()
client = OpenAI(api_key=OPENAI_KEY)  

class ChatIn(BaseModel):
    message: str
    history: list[dict] = []

app = FastAPI(title="PetroAgent Chat Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

SYSTEM_PROMPT = (
    "You are PetroAgent—an expert petroleum engineering copilot. "
    "You can explain drilling, completions, reservoir, production, petrophysics, "
    "and also help users upload CSV data, define features/targets, train ML models "
    "(NeuralNet, GaussianProcess, RandomForest, XGBoost, Transformer), analyze plots, "
    "and save artifacts. Be clear, accurate, and proactive."
)

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/chat")
def chat(payload: ChatIn):
    msgs = [{"role":"system","content":SYSTEM_PROMPT}]
    # map history [{"role": "user"/"assistant", "content": "..."}]
    for m in payload.history:
        if m.get("role") in ("user","assistant","system") and m.get("content"):
            msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role":"user","content": payload.message})

    try:
        # gpt-4o-mini is a great low-latency pick; change to gpt-4.1 if you want heavier reasoning.
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0.3,
        )
        answer = resp.choices[0].message.content
        return {"reply": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
