import os, joblib, numpy as np
from .base import train_val_split, regression_metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def train_gp(X, y, out_dir: str):
    Xtr, Xte, ytr, yte = train_val_split(X, y)
    kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
    model.fit(Xtr, ytr)
    yp, _ = model.predict(Xte, return_std=True)
    metrics = regression_metrics(yte, yp)
    model_path = os.path.join(out_dir, "gp_model.joblib")
    joblib.dump(model, model_path)
    return {"metrics": metrics, "y_true": yte.tolist(), "y_pred": yp.tolist(), "model_path": model_path}
