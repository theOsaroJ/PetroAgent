import os
import uuid
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from trainers import utils, rf, gp, xgb, nn, transformer, plots, save_load

DATA_DIR = "/app/data"
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = FastAPI(title="PetroAgent ML Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "service": "ml_service"}

class SchemaOut(BaseModel):
    upload_id: str
    n_rows: int
    n_cols: int
    columns: list[str]
    dtypes: dict

@app.post("/ml/upload", response_model=SchemaOut)
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in [".csv"]:
        return {"error": "Only CSV supported for now."}
    uid = str(uuid.uuid4())
    save_path = os.path.join(UPLOADS_DIR, f"{uid}.csv")
    with open(save_path, "wb") as f:
        f.write(await file.read())
    df = pd.read_csv(save_path)
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    return SchemaOut(upload_id=uid, n_rows=len(df), n_cols=df.shape[1], columns=df.columns.tolist(), dtypes=dtypes)

class TrainIn(BaseModel):
    upload_id: str
    target: str
    features: list[str]
    task: str                      # "regression" | "classification"
    model_type: str                # "rf" | "gp" | "xgb" | "nn" | "transformer"
    output_dir: str                # absolute path inside container or relative under /app/data/outputs
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    standardize: bool = True
    max_rows: int | None = None    # subsample if needed for GP

class TrainOut(BaseModel):
    model_path: str
    metrics: dict
    artifacts: list[str]

@app.post("/ml/train", response_model=TrainOut)
def train(payload: TrainIn):
    csv_path = os.path.join(UPLOADS_DIR, f"{payload.upload_id}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Upload ID not found.")
    out_dir = payload.output_dir.strip() or OUTPUTS_DIR
    if not out_dir.startswith("/"):
        out_dir = os.path.join(OUTPUTS_DIR, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if payload.max_rows and len(df) > payload.max_rows:
        df = df.sample(payload.max_rows, random_state=payload.random_state)

    X, y, meta = utils.prepare_xy(
        df=df,
        features=payload.features,
        target=payload.target,
        task=payload.task,
        standardize=payload.standardize,
        for_transformer=(payload.model_type == "transformer"),
    )

    splits = utils.split_data(
        X, y, val_size=payload.val_size, test_size=payload.test_size, random_state=payload.random_state
    )

    if payload.model_type == "rf":
        model, metrics = rf.train_rf(splits, task=payload.task)
    elif payload.model_type == "gp":
        model, metrics = gp.train_gp(splits, task=payload.task)
    elif payload.model_type == "xgb":
        model, metrics = xgb.train_xgb(splits, task=payload.task)
    elif payload.model_type == "nn":
        model, metrics = nn.train_nn(splits, task=payload.task, meta=meta)
    elif payload.model_type == "transformer":
        model, metrics = transformer.train_tabtransformer(splits, task=payload.task, meta=meta)
    else:
        raise ValueError("Unknown model_type")

    model_path = save_load.save_model(model, meta, out_dir, payload.model_type)
    art_paths = plots.generate_artifacts(splits, model, meta, out_dir, payload.task, payload.model_type)

    return TrainOut(model_path=model_path, metrics=metrics, artifacts=art_paths)
