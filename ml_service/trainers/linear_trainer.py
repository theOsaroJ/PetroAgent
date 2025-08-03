import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_linear(data_path: str):
    df = pd.read_csv(data_path)
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "linear.model")

def predict_linear(model_path: str, input_data: list[float]):
    model = joblib.load(model_path)
    return model.predict([input_data])[0]
