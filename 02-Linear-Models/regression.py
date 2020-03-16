import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


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
                residual = self._Y - self._X.dot(self.beta_hat)
                mse = (1 / self._N) * residual.T.dot(residual)
                training_loss.append(mse)
                self.beta_hat -= 2 * self.learning_rate * (-residual.T.dot(self._X))
            return training_loss
        else:
            self.beta_hat = np.linalg.inv(self._X.T.dot(self._X)).dot(self._X.T).dot(self._Y)
            residual = self._Y - self._X.dot(self.beta_hat)
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
                residual = self._Y - self._X.dot(self.beta_hat)
                mse = (1 / self._N) * (residual.T.dot(residual) + self.penalty * np.sum(self.beta_hat ** 2))
                training_loss.append(mse)
                self.beta_hat -= 2 * self.learning_rate * (-residual.T.dot(self._X) + self.penalty * self.beta_hat)
            return training_loss
        else:
            self.beta_hat = np.linalg.inv(self._X.T.dot(self._X) + self.penalty * np.eye(self._p)).dot(self._X.T).dot(self._Y)
            residual = self._Y - self._X.dot(self.beta_hat)
            loss = (1 / self._N) * (residual.T.dot(residual) + self.penalty * np.sum(self.beta_hat ** 2))
            return loss


class LassoRegression:
    def __init__(self, n_iter=10000, penalty=10, intercept=False):
        self.n_iter = n_iter
        self.beta_hat = []
        self.intercept = intercept
        self.penalty = penalty
        self._X = None
        self._Y = None
        self._p = None
        self._N = None

    def _preprocess_data(self, X, Y):
        self._X = np.asarray(X)
        self._Y = np.asarray(Y)
        self._p = X.shape[1]
        self._N = X.shape[0]

        # Tricky way: here we do not divide the standard deviation of each features
        # Instead, we divide the l2-norm as sklearn does
        # For our data, the l2-norm did make a descent trend in loss
        # If we used np.std, the loss would be constant(hard to tune lambda later)
        self._X = self._X / np.linalg.norm(self._X, axis=0)

        if self.intercept:
            ones = np.reshape(np.ones(self._N), (self._N, 1))
            self._X = np.hstack((ones, self._X))
            self._p = self._p + 1
        self.beta_hat = np.ones(shape=self._p)

    def _soft_threshold(self, ols, lambdap):
        if ols < -lambdap / 2:
            return ols + lambdap
        elif ols > lambdap / 2:
            return ols - lambdap
        else:
            return 0

    def fit(self, X, Y):
        # solve it with coordinate descent
        self._preprocess_data(X, Y)
        training_loss = []
        betas = []
        for i in range(self.n_iter):
            for j in range(self._p):
                if not (self.intercept and j == 0):
                    X_j = self._X[:, j]
                    _rio_j = np.reshape(X_j, (1, self._N)).dot(self._Y-self._X.dot(self.beta_hat) + self.beta_hat[j]*X_j)
                    self.beta_hat[j] = self._soft_threshold(_rio_j, self.penalty)
            residual = self._Y - self._X.dot(self.beta_hat)
            mse = (1 / self._N) * (residual.T.dot(residual) + self.penalty * np.sum(np.abs(self.beta_hat)))
            training_loss.append(mse)
            betas.append(deepcopy(self.beta_hat))
        return training_loss, betas

    def predict(self, X_new):
        return X_new.dot(self.beta_hat)

    def plot_cv_lambdas(self, X, Y, lambdas, labels):
        beta_lst = []
        for lamb in lambdas:
            l = LassoRegression(penalty=lamb, n_iter=100)
            _,_ = l.fit(X, Y)
            beta_lst.append(l.beta_hat)
        beta_lasso = np.stack(beta_lst).T

        # Plot results
        n, _ = beta_lasso.shape
        plt.figure(figsize=(12, 8))

        for i in range(n):
            plt.plot(lambdas, beta_lasso[i], label=labels[i])

        plt.xscale('log')
        plt.xlabel('Log($\\lambda$)')
        plt.ylabel('Coefficients')
        plt.title('Lasso Paths - Numpy implementation')
        plt.legend()
        plt.axis('tight')