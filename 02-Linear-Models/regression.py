import numpy as np


class LinearRegression:
    def __init__(self, n_iter=10000, learning_rate=1e-6, gradient_descent=True, intercept=False):
        self.gradient_descent = gradient_descent
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.beta_hat = []
        self.intercept = intercept
        self._X = None
        self._Y = None
        self._p = None
        self._N = None

    def _preprocess_data(self, X, Y):
        self._X = np.asarray(X)
        self._Y = np.asarray(Y)
        self._p = X.shape[1]
        self._N = X.shape[0]

        if self.intercept:
            ones = np.reshape(np.ones(self._N), (self._N, 1))
            self._X = np.hstack((ones, self._X))
            self._p = self._p + 1
        self.beta_hat = np.zeros(self._p)

    def fit(self, X, Y):
        self._preprocess_data(X, Y)
        if self.gradient_descent:
            training_loss = []
            self.beta_hat = np.random.uniform(size=self._p)
            for i in range(self.n_iter):
                residual = Y - X.dot(self.beta_hat)
                mse = (1 / self._N) * residual.T.dot(residual)
                training_loss.append(mse)
                self.beta_hat -= 2 * self.learning_rate * (-residual.T.dot(X))
            return training_loss
        else:
            self.beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
            residual = Y - X.dot(self.beta_hat)
            mse = (1 / self._N) * residual.T.dot(residual)
            return mse

    def predict(self, X_new):
        return X_new.dot(self.beta_hat)


class RidgeRegression(LinearRegression):
    def __init__(self, n_iter=10000, learning_rate=1e-6, penalty=0.1, gradient_descent=True, intercept=False):
        super().__init__(n_iter, learning_rate, gradient_descent, intercept)
        self.penalty = penalty

    def fit(self, X, Y):
        super()._preprocess_data(X, Y)
        if self.gradient_descent:
            training_loss = []
            self.beta_hat = np.random.uniform(size=self._p)
            for i in range(self.n_iter):
                residual = Y - X.dot(self.beta_hat)
                mse = (1 / self._N) * (residual.T.dot(residual) + self.penalty * np.sum(self.beta_hat ** 2))
                training_loss.append(mse)
                self.beta_hat -= 2 * self.learning_rate * (-residual.T.dot(X) + self.penalty * self.beta_hat)
            return training_loss
        else:
            self.beta_hat = np.linalg.inv(X.T.dot(X) + self.penalty * np.eye(self._p)).dot(X.T).dot(Y)
            residual = Y - X.dot(self.beta_hat)
            loss = (1 / self._N) * (residual.T.dot(residual) + self.penalty * np.sum(self.beta_hat ** 2))
            return loss
