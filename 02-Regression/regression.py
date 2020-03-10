import numpy as np


class LinearRegression:
    def __init__(self, n_iter=10000, learning_rate=1e-6, gradient_descent=True, intercept=False):
        self.gradient_descent = gradient_descent
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.beta_hat = []
        self.intercept = intercept

    def fit(self, X, Y):

        X = np.asarray(X)
        Y = np.asarray(Y)
        p = X.shape[1]
        N = X.shape[0]

        if self.intercept:
            ones = np.reshape(np.ones(N),(N,1))
            X = np.hstack((ones, X))
            p = p + 1
        self.beta_hat = np.zeros(p)

        if self.gradient_descent:
            training_loss = []
            self.beta_hat = np.random.uniform(size=p)
            for i in range(self.n_iter):
                residual = Y - X.dot(self.beta_hat)
                mse = (1 / N) * residual.T.dot(residual)
                training_loss.append(mse)
                self.beta_hat -= 2 * self.learning_rate * (-residual.T.dot(X))
            return training_loss
        else:
            self.beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
            residual = Y - X.dot(self.beta_hat)
            mse = (1 / N) * residual.T.dot(residual)
            return mse

    def predict(self, X_new):
        return X_new.dot(self.beta_hat)
