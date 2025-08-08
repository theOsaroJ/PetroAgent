from sklearn.ensemble import RandomForestRegressor

def train(X, y):
    model = RandomForestRegressor(n_estimators=200, max_depth=10)
    model.fit(X, y)
    return model
