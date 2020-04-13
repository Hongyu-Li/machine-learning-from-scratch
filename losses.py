import numpy as np

from activations import Sigmoid


class MSE:
    def __call__(self, y, y_pred):
        return np.mean((y-y_pred)**2)

    def gradient(self, y, y_pred):
        return -(y-y_pred)

    def hessian(self, y, y_pred):
        return np.ones_like(y)


class LogLoss:
    def __init__(self):
        self.sigmoid = Sigmoid()

    def __call__(self, y, y_pred):
        p = self.sigmoid(y_pred)
        # avoid overflow and underflow issue
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return y * np.log2(p) + (1-y) * np.log2(1-p)

    def gradient(self, y, y_pred):
        p = self.sigmoid(y_pred)
        return -(y - p)

    def hessian(self, y, y_pred):
        p = self.sigmoid(y_pred)
        return p * (1-p)