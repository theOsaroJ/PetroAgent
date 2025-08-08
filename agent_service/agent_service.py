from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import openai, os

openai.api_key = open(os.getenv("OPENAI_API_KEY"),"r").read().strip()

app = FastAPI(title="PetroAgent Chat Service")
app.mount("/static", StaticFiles(directory="public"), name="static")

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    # just echo file info back
    return {"filename": file.filename, "content_type": file.content_type}

@app.post("/api/chat")
async def chat(prompt: str = Form(...)):
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}]
    )
    return {"reply": res.choices[0].message.content}
