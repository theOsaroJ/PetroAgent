import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BACKEND_URL = os.getenv("BACKEND_URL")
ML_URL = os.getenv("ML_URL")

app = FastAPI()
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

def upload_csv_tool(file_path: str):
    with open(file_path, "rb") as f:
        files = {"file": f}
        resp = requests.post(f"{BACKEND_URL}/upload", files=files)
    return resp.json()

tools = [
    Tool(
        name="upload_csv",
        func=upload_csv_tool,
        description="Use this tool to upload a CSV and detect headers."
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True
)

class ChatRequest(BaseModel):
    text: str

@app.post("/chat")
def chat(req: ChatRequest):
    result = agent.run(req.text)
    return {"response": result}
