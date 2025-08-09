from sklearn.ensemble import RandomForestRegressor

def train_rf(X_train, y_train):
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    return rf
