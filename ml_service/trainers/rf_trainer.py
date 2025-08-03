from sklearn.ensemble import RandomForestRegressor
import pandas as pd

model_rf = None

def train_rf(df, inputs, target):
    global model_rf
    X = df[inputs].values
    y = df[target].values
    model = RandomForestRegressor()
    model.fit(X, y)
    model_rf = model
    return model

def predict_rf(df):
    return model_rf.predict(df.values)
