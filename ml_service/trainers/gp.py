import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

def train_gp(splits, task="regression"):
    Xtr, Xval, Xte = splits["X_train"], splits["X_val"], splits["X_test"]
    ytr, yval, yte = splits["y_train"], splits["y_val"], splits["y_test"]
    kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
    if task == "regression":
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42, alpha=1e-6)
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        metrics = {"MAE": float(mean_absolute_error(yte, preds)), "R2": float(r2_score(yte, preds))}
    else:
        # For large classes GP can be heavy; sampling assumed upstream if needed
        model = GaussianProcessClassifier(kernel=kernel, random_state=42, max_iter_predict=200)
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        metrics = {"ACC": float(accuracy_score(yte, preds)), "F1": float(f1_score(yte, preds, average="macro"))}
    return model, metrics
