from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection    import train_test_split

def train_gp(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    gp = GaussianProcessRegressor()
    gp.fit(X_tr, y_tr)
    return gp, gp.score(X_te, y_te)
