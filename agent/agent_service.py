import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

def load_openai_key() -> str:
    # Priority: env var, secret file, local api_key.txt
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    secret_path = "/run/secrets/openai_api_key"
    if os.path.exists(secret_path):
        return open(secret_path, "r", encoding="utf-8").read().strip()
    local_path = os.path.join(os.path.dirname(__file__), "api_key.txt")
    if os.path.exists(local_path):
        return open(local_path, "r", encoding="utf-8").read().strip()
    raise RuntimeError("OpenAI API key not found. Provide agent/api_key.txt or set OPENAI_API_KEY.")

OPENAI_KEY = load_openai_key()
# Important: do NOT pass unsupported kwargs like 'proxies'
client = OpenAI(api_key=OPENAI_KEY)

app = FastAPI(title="PetroAgent Chat Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class ChatIn(BaseModel):
    message: str

class ChatOut(BaseModel):
    reply: str

@app.get("/health")
def health():
    return {"status": "ok", "service": "agent"}

@app.post("/api/chat", response_model=ChatOut)
def chat(payload: ChatIn):
    # Use a strong but cost-effective model; you can swap for gpt-4o/gpt-4.1 etc.
    # See openai-python docs: https://github.com/openai/openai-python
    prompt = payload.message.strip()
    if not prompt:
        return ChatOut(reply="Please enter a question.")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You are PetroAgent: a petroleum engineering + ML assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    text = resp.choices[0].message.content or ""
    return ChatOut(reply=text)
