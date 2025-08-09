import os, json, joblib, datetime

def save_model(model, meta, out_dir, model_type):
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(out_dir, f"model_{model_type}_{ts}.joblib")
    joblib.dump({"model": model, "meta": meta}, model_path)
    with open(os.path.join(out_dir, f"meta_{model_type}_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return model_path
