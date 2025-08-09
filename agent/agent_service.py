import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# 1) Read key from file (default: /run/secrets/openai_api_key) or env var
OPENAI_KEY_FILE = os.getenv("OPENAI_KEY_FILE", "api_key.txt")
OPENAI_API_KEY = ""
if os.path.exists(OPENAI_KEY_FILE):
    with open(OPENAI_KEY_FILE, "r") as f:
        OPENAI_API_KEY = f.read().strip()
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("OpenAI API key missing: set OPENAI_API_KEY or mount OPENAI_KEY_FILE")

client = OpenAI(api_key=OPENAI_API_KEY)  # no proxies kwarg

app = FastAPI()

class ChatIn(BaseModel):
    message: str

@app.post("/api/chat")
def chat(body: ChatIn):
    try:
        # use the latest model you’re allowed to use in your account, e.g. gpt-4o-mini
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are PetroAgent."},
                      {"role":"user","content":body.message}],
            temperature=0.2,
        )
        return {"reply": resp.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
