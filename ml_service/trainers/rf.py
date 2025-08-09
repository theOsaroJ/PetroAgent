from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

def train_rf(splits, task="regression"):
    Xtr, Xval, Xte = splits["X_train"], splits["X_val"], splits["X_test"]
    ytr, yval, yte = splits["y_train"], splits["y_val"], splits["y_test"]
    if task == "regression":
        model = RandomForestRegressor(n_estimators=400, max_features="sqrt", random_state=42, n_jobs=-1)
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        metrics = {"MAE": float(mean_absolute_error(yte, preds)), "R2": float(r2_score(yte, preds))}
    else:
        model = RandomForestClassifier(n_estimators=400, max_features="sqrt", random_state=42, n_jobs=-1)
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        metrics = {"ACC": float(accuracy_score(yte, preds)), "F1": float(f1_score(yte, preds, average="macro"))}
    return model, metrics
