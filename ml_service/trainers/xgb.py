from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

def train_xgb(splits, task="regression"):
    Xtr, Xval, Xte = splits["X_train"], splits["X_val"], splits["X_test"]
    ytr, yval, yte = splits["y_train"], splits["y_val"], splits["y_test"]

    if task == "regression":
        model = XGBRegressor(
            n_estimators=800, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.0, reg_lambda=1.0, random_state=42, n_jobs=-1, tree_method="hist"
        )
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        preds = model.predict(Xte)
        metrics = {"MAE": float(mean_absolute_error(yte, preds)), "R2": float(r2_score(yte, preds))}
    else:
        model = XGBClassifier(
            n_estimators=800, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.0, reg_lambda=1.0, random_state=42, n_jobs=-1, tree_method="hist", eval_metric="mlogloss"
        )
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        preds = model.predict(Xte)
        metrics = {"ACC": float(accuracy_score(yte, preds)), "F1": float(f1_score(yte, preds, average="macro"))}
    return model, metrics
