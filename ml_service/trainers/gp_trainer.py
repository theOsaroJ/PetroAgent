from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

def train_gp(X_train, y_train):
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=1.0) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True, random_state=42)
    gp.fit(X_train, y_train)
    return gp
