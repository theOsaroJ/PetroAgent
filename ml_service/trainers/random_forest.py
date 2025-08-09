import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
from .utils import metric_dict, plot_pred_vs_actual, plot_residuals
import numpy as np
import matplotlib.pyplot as plt

def train_random_forest(X, y, save_dir, test_size=0.2, random_state=42, n_estimators=600, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    yp = model.predict(X_val)

    metrics = metric_dict(y_val.values, yp)

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "model_random_forest.joblib")
    dump(model, model_path)

    p1 = os.path.join(save_dir, "pred_vs_actual_rf.png")
    p2 = os.path.join(save_dir, "residuals_rf.png")
    plot_pred_vs_actual(y_val.values, yp, p1)
    plot_residuals(y_val.values, yp, p2)

    # feature importance
    if hasattr(model, "feature_importances_"):
        p3 = os.path.join(save_dir, "feature_importance_rf.png")
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1]
        plt.figure()
        plt.bar(range(len(idx)), importances[idx])
        plt.xticks(range(len(idx)), np.array(X.columns)[idx], rotation=45, ha="right")
        plt.title("Feature Importance (RF)")
        plt.tight_layout()
        plt.savefig(p3, dpi=160)
        plt.close()
        artifacts = [p1, p2, p3]
    else:
        artifacts = [p1, p2]

    return {"metrics": metrics, "artifacts": artifacts, "model_path": model_path, "notes": "RF with strong baseline settings."}
