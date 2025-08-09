import os
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from joblib import dump
from .utils import metric_dict, plot_pred_vs_actual, plot_residuals

def train_gp(X, y, save_dir, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=random_state)
    model.fit(X_train, y_train)
    yp = model.predict(X_val)

    metrics = metric_dict(y_val.values, yp)

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "model_gp.joblib")
    dump(model, model_path)

    p1 = os.path.join(save_dir, "pred_vs_actual_gp.png")
    p2 = os.path.join(save_dir, "residuals_gp.png")
    plot_pred_vs_actual(y_val.values, yp, p1)
    plot_residuals(y_val.values, yp, p2)

    return {"metrics": metrics, "artifacts": [p1, p2], "model_path": model_path, "notes": "Gaussian Process (RBF + noise)."}
