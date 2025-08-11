import os, joblib
from .base import train_val_split, regression_metrics
from sklearn.ensemble import RandomForestRegressor

def train_rf(X, y, out_dir: str):
    Xtr, Xte, ytr, yte = train_val_split(X, y)
    model = RandomForestRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    metrics = regression_metrics(yte, yp)
    model_path = os.path.join(out_dir, "rf_model.joblib")
    joblib.dump(model, model_path)
    return {"metrics": metrics, "y_true": yte.tolist(), "y_pred": yp.tolist(), "model_path": model_path}
