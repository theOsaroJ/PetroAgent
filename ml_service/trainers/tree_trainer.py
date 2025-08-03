from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_tree(model_type, X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    if model_type == "random_forest":
        m = RandomForestRegressor()
    elif model_type == "xgboost":
        m = xgb.XGBRegressor()
    elif model_type == "linear":
        m = LinearRegression()
    else:
        raise ValueError("Unknown tree model_type")
    m.fit(X_tr, y_tr)
    return m, m.score(X_te, y_te)
