import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def split_xy(df: pd.DataFrame, feats, target):
    X = df[feats].values
    y = df[target].values
    return X, y

def save_metrics_json(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

def parity_plot(y_true, y_pred, out_path: str):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Parity Plot")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
