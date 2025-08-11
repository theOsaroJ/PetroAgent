import os
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from openai import OpenAI

app = FastAPI(title="PetroAgent - Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_KEY_FILE = os.getenv("OPENAI_API_KEY_FILE")
OPENAI_KEY = None
if OPENAI_KEY_FILE and os.path.exists(OPENAI_KEY_FILE):
    with open(OPENAI_KEY_FILE, "r") as f:
        OPENAI_KEY = f.read().strip()
else:
    OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_KEY)  # modern SDK (no proxies kwarg)

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml_service:8000")

class ChatIn(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(payload: ChatIn):
    # lightweight system prompt
    msgs = [
        {"role": "system", "content": "You are PetroAgent, an expert petroleum data copilot."},
        {"role": "user", "content": payload.message},
    ]
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=msgs,
        temperature=0.3,
    )
    return {"reply": resp.choices[0].message.content}

@app.post("/api/columns")
async def detect_columns(file: UploadFile = File(...)):
    async with httpx.AsyncClient(timeout=120) as x:
        files = {"file": (file.filename, await file.read(), file.content_type or "text/csv")}
        r = await x.post(f"{ML_SERVICE_URL}/columns", files=files)
        r.raise_for_status()
        return r.json()

@app.post("/api/upload")
async def upload_and_train(
    file: UploadFile = File(...),
    features: str = Form(...),
    target: str = Form(...),
    model_type: str = Form(...),    # rf | xgb | gp | nn | transformer
    save_dir: Optional[str] = Form(None)
):
    async with httpx.AsyncClient(timeout=None) as x:
        files = {"file": (file.filename, await file.read(), file.content_type or "text/csv")}
        data = {"features": features, "target": target, "model_type": model_type}
        if save_dir: data["save_dir"] = save_dir
        r = await x.post(f"{ML_SERVICE_URL}/train", files=files, data=data)
        r.raise_for_status()
        return r.json()

from fastapi.responses import FileResponse
from fastapi import Query

@app.get("/api/static")
async def static_proxy(path: str = Query(...)):
    if not os.path.exists(path): 
        raise HTTPException(404, "File not found")
    return FileResponse(path)
