import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

app = FastAPI()

class Prompt(BaseModel):
    message: str

@app.post("/chat")
def chat(req: Prompt):
    try:
        agent = initialize_agent([], llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=False)
        return {"reply": agent.run(req.message)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
