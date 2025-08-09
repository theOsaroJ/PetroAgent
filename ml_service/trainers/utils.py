import os
import numpy as np
import matplotlib.pyplot as plt

def ensure_subdir(path: str):
    os.makedirs(path, exist_ok=True)

def sanitize_rel(rel: str) -> str:
    rel = rel.strip().lstrip("/").replace("..", "")
    return rel or "artifacts/run1"

def metric_dict(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def plot_pred_vs_actual(y_true, y_pred, out_path):
    plt.figure()
    plt.scatter(y_true, y_pred, s=10)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_residuals(y_true, y_pred, out_path):
    resid = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, resid, s=10)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residuals")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def save_plot_paths(paths, data_root):
    rels = []
    for p in paths:
        rels.append(os.path.relpath(p, data_root).replace("\\", "/"))
    return rels
