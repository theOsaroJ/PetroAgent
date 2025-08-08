import xgboost as xgb

def train(X, y):
    model = xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05)
    model.fit(X, y)
    return model
