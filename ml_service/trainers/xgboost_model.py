import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from joblib import dump
from .utils import metric_dict, plot_pred_vs_actual, plot_residuals
import numpy as np
import matplotlib.pyplot as plt

def train_xgboost(X, y, save_dir, test_size=0.2, random_state=42, n_estimators=700, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=0,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    yp = model.predict(X_val)
    metrics = metric_dict(y_val.values, yp)

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "model_xgb.joblib")
    dump(model, model_path)

    p1 = os.path.join(save_dir, "pred_vs_actual_xgb.png")
    p2 = os.path.join(save_dir, "residuals_xgb.png")
    plot_pred_vs_actual(y_val.values, yp, p1)
    plot_residuals(y_val.values, yp, p2)

    # feature importance
    try:
      importances = model.feature_importances_
      idx = np.argsort(importances)[::-1]
      p3 = os.path.join(save_dir, "feature_importance_xgb.png")
      plt.figure()
      plt.bar(range(len(idx)), importances[idx])
      plt.xticks(range(len(idx)), np.array(X.columns)[idx], rotation=45, ha="right")
      plt.title("Feature Importance (XGB)")
      plt.tight_layout()
      plt.savefig(p3, dpi=160)
      plt.close()
      arts = [p1, p2, p3]
    except Exception:
      arts = [p1, p2]

    return {"metrics": metrics, "artifacts": arts, "model_path": model_path, "notes": "XGBoost tuned for tabular regression."}
