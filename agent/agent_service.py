from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import requests
from pathlib import Path

# read key from mounted file
API_KEY = Path(os.getenv("OPENAI_API_KEY_FILE", "api_key.txt")).read_text().strip()

from langchain_openai import ChatOpenAI
from langchain import LLMChain, PromptTemplate

llm = ChatOpenAI(temperature=0, openai_api_key=API_KEY)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(req: ChatRequest):
    template = "You are an AI assistant.\nUser: {input}\nAssistant:"
    prompt = PromptTemplate(input_variables=["input"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)
    resp = chain.run(input=req.message)
    return {"response": resp}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    return {"columns": list(df.columns), "n_rows": len(df)}

@app.post("/api/train")
async def train(body: dict):
    return requests.post(
        f"http://ml_service:8000/train/{body['model_type']}",
        json={
          "data": body["data"],
          "input_columns": body["input_columns"],
          "target_column": body["target_column"]
        }
    ).json()

@app.post("/api/predict")
async def predict(body: dict):
    return requests.post(
        f"http://ml_service:8000/predict/{body['model_type']}",
        json={
          "data": body["data"],
          "input_columns": body["input_columns"]
        }
    ).json()
