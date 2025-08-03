from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pandas as pd

model_gp = None

def train_gp(df, inputs, target):
    global model_gp
    X = df[inputs].values
    y = df[target].values
    kernel = C(1.0, (1e-3,1e3)) * RBF(1.0, (1e-3,1e3))
    model = GaussianProcessRegressor(kernel=kernel)
    model.fit(X, y)
    model_gp = model
    return model

def predict_gp(df):
    return model_gp.predict(df.values)
