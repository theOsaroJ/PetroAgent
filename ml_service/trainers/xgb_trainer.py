from xgboost import XGBRegressor

def train_xgb(X_train, y_train, X_val, y_val):
    xgb = XGBRegressor(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return xgb
