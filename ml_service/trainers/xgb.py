import os, joblib
from .base import train_val_split, regression_metrics
from xgboost import XGBRegressor

def train_xgb(X, y, out_dir: str):
    Xtr, Xte, ytr, yte = train_val_split(X, y)
    model = XGBRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    yp = model.predict(Xte)
    metrics = regression_metrics(yte, yp)
    model_path = os.path.join(out_dir, "xgb_model.joblib")
    joblib.dump(model, model_path)  # xgboost model is picklable
    return {"metrics": metrics, "y_true": yte.tolist(), "y_pred": yp.tolist(), "model_path": model_path}
