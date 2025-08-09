from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import httpx

AGENT_URL = os.getenv("AGENT_URL", "http://agent:7000")
ML_URL = os.getenv("ML_URL", "http://ml_service:8000")

app = FastAPI(title="PetroAgent Backend Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/api/health")
async def health():
    return {"ok": True}

# ----- Chat -----
@app.post("/api/chat")
async def chat(payload: dict):
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{AGENT_URL}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()

# ----- Upload -----
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    files = {"file": (file.filename, await file.read(), file.content_type or "text/csv")}
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{ML_URL}/api/upload", files=files)
        r.raise_for_status()
        return r.json()

# ----- List models -----
@app.get("/api/models")
async def models():
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(f"{ML_URL}/api/models")
        r.raise_for_status()
        return r.json()

# ----- Train -----
@app.post("/api/train")
async def train(payload: dict):
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{ML_URL}/api/train", json=payload)
        r.raise_for_status()
        return r.json()

# ----- Plots / describe -----
@app.post("/api/plots")
async def plots(payload: dict):
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{ML_URL}/api/plots", json=payload)
        r.raise_for_status()
        return r.json()

@app.post("/api/describe")
async def describe(payload: dict):
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{ML_URL}/api/describe", json=payload)
        r.raise_for_status()
        return r.json()
