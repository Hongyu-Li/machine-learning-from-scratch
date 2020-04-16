import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


class BayesianLinearRegression:
    """
    Y ~ N(\beta.T X, sigma^2) given sigma^2 is known
    prior: w|X ~ N(0, \Sigma_p^2)
    Ref: https://www.cnblogs.com/nxf-rabbit75/p/10382368.html
    https://github.com/tdomhan/pyblr/blob/master/blr/blr.py

    """

    def __init__(self, error_std=1, prior_w_std=None):
        self.error_std = error_std
        self.prior_w_std = prior_w_std
        self.posterior_std, self.posterior_mean = None, None

    def fit(self, X, y):
        n, p = X.shape
        if self.prior_w_std is None:
            self.prior_w_std = np.eye(p)
        gamma = (1 / self.error_std ** 2) * X.T.dot(X) + np.linalg.inv(self.prior_w_std) ** 2
        self.posterior_std = np.linalg.inv(gamma)
        self.posterior_mean = (1 / self.error_std ** 2) * self.posterior_std.dot(X.T).dot(y)
        return [self.predict(x) for x in X]

    def predict(self, X_new):
        return X_new.dot(self.posterior_mean)

# if __name__ == '__main__':
#     X, y = make_regression(n_samples=100, n_features=2, random_state=0, noise=5)
#     model = BayesianLinearRegression(error_std=5, prior_w_std=np.eye(2)*5)
#     preds = model.fit(X, y)
#     plt.scatter(range(len(X)), y, c='blue')
#     plt.scatter(range(len(X)), preds, c='red')
#     plt.show()
