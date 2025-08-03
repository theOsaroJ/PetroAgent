import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_rf(data_path: str):
    df = pd.read_csv(data_path)
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, "rf.model")

def predict_rf(model_path: str, input_data: list[float]):
    model = joblib.load(model_path)
    return model.predict([input_data])[0]
