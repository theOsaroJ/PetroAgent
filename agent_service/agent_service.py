import os, requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

# Read key
with open('api_key.txt') as f:
    OPENAI_API_KEY = f.read().strip()
if not OPENAI_API_KEY:
    raise ValueError("api_key.txt is empty")

llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
BACKEND = os.getenv("BACKEND_URL", "http://backend:5000/api")

class Req(BaseModel):
    message: str

upload_tool = Tool(
    name="upload_csv",
    description="Upload CSV; input is path",
    func=lambda p: requests.post(f"{BACKEND}/upload", files={"file": open(p,"rb")}).json()
)
train_tool = Tool(
    name="train_model",
    description="Train: model_type, file_path, target_column, feature_columns",
    func=lambda p: requests.post(
        f"{BACKEND}/train/{p['model_type']}",
        json={"filePath":p["file_path"],"targetColumn":p["target_column"],"featureColumns":p["feature_columns"]}
    ).json()
)
predict_tool = Tool(
    name="predict_model",
    description="Predict: model_type, input_data",
    func=lambda p: requests.post(
        f"{BACKEND}/predict/{p['model_type']}",
        json={"inputData":p["input_data"]}
    ).json()
)

agent = initialize_agent(
    [upload_tool, train_tool, predict_tool],
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True
)

app = FastAPI()

@app.post("/agent/respond")
async def respond(req: Req):
    try:
        return {"response": agent.run(req.message)}
    except Exception as e:
        raise HTTPException(500, detail=str(e))
