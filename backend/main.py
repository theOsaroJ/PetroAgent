from fastapi import FastAPI

app = FastAPI(title="PetroAgent Backend")

@app.get("/health")
def health():
    return {"status": "ok"}
