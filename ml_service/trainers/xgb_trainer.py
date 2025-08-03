import pandas as pd
import xgboost as xgb
import joblib

def train_xgboost(data_path: str):
    df = pd.read_csv(data_path)
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    model = xgb.XGBRegressor(n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, "xgboost.model")

def predict_xgboost(model_path: str, input_data: list[float]):
    model = joblib.load(model_path)
    return model.predict([input_data])[0]
