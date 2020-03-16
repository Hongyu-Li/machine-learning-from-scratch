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
        self.__X = None
        self.__Y = None
        self.__p = None
        self.__N = None

    def __preprocess_data(self, X, Y):
        self.__X = np.asarray(X)
        self.__Y = np.asarray(Y)
        self.__p = X.shape[1]
        self.__N = X.shape[0]

        if self.intercept:
            ones = np.reshape(np.ones(self.__N), (self.__N, 1))
            self.__X = np.hstack((ones, self.__X))
            self.__p = self.__p + 1
        self.beta_hat = np.zeros(self.__p)

    def fit(self, X, Y):
        self.__preprocess_data(X, Y)
        if self.gradient_descent:
            training_loss = []
            self.beta_hat = np.random.uniform(size=self.__p)
            for i in range(self.n_iter):
                residual = self.__Y - self.__X.dot(self.beta_hat)
                mse = (1 / self.__N) * residual.T.dot(residual)
                training_loss.append(mse)
                self.beta_hat -= 2 * self.learning_rate * (-residual.T.dot(self.__X))
            return training_loss
        else:
            self.beta_hat = np.linalg.inv(self.__X.T.dot(self.__X)).dot(self.__X.T).dot(self.__Y)
            residual = self.__Y - self.__X.dot(self.beta_hat)
            mse = (1 / self.__N) * residual.T.dot(residual)
            return mse

    def predict(self, X_new):
        return X_new.dot(self.beta_hat)


class RidgeRegression(LinearRegression):
    def __init__(self, n_iter=10000, learning_rate=1e-6, penalty=0.1, gradient_descent=True, intercept=False):
        super().__init__(n_iter, learning_rate, gradient_descent, intercept)
        self.penalty = penalty

    def fit(self, X, Y):
        super().__preprocess_data(X, Y)

        if self.gradient_descent:
            training_loss = []
            self.beta_hat = np.random.uniform(size=self.__p)
            for i in range(self.n_iter):
                residual = self.__Y - self.__X.dot(self.beta_hat)
                mse = (1 / self.__N) * (residual.T.dot(residual) + self.penalty * np.sum(self.beta_hat ** 2))
                training_loss.append(mse)
                self.beta_hat -= 2 * self.learning_rate * (-residual.T.dot(self.__X) + self.penalty * self.beta_hat)
            return training_loss
        else:
            self.beta_hat = np.linalg.inv(self.__X.T.dot(self.__X) + self.penalty * np.eye(self.__p)).dot(self.__X.T).dot(self.__Y)
            residual = self.__Y - self.__X.dot(self.beta_hat)
            loss = (1 / self.__N) * (residual.T.dot(residual) + self.penalty * np.sum(self.beta_hat ** 2))
            return loss


class LassoRegression:
    def __init__(self, n_iter=10000, penalty=10, intercept=False):
        self.n_iter = n_iter
        self.beta_hat = []
        self.intercept = intercept
        self.__X = None
        self.__Y = None
        self.__p = None
        self.__N = None

    def _preprocess_data(self, X, Y):
        self.__X = np.asarray(X)
        self.__Y = np.asarray(Y)
        self.__p = X.shape[1]
        self.__N = X.shape[0]

        # Tricky way: here we do not divide the standard deviation of each features
        # Instead, we divide the l2-norm as sklearn does
        # For our data, the l2-norm did make a descent trend in loss
        # If we used np.std, the loss would be constant(hard to tune lambda later)
        self.__X = self.__X / np.linalg.norm(self.__X, axis=0)

        if self.intercept:
            ones = np.reshape(np.ones(self.__N), (self.__N, 1))
            self.__X = np.hstack((ones, self.__X))
            self.__p = self.__p + 1
        self.beta_hat = np.ones(shape=self.__p)

    def __soft_threshold(self, ols, lambdap):
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
            for j in range(self.__p):
                if not (self.intercept and j == 0):
                    X_j = self.__X[:, j]
                    __rio_j = np.reshape(X_j, (1, self.__N)).dot(self.__Y-self.__X.dot(self.beta_hat) + self.beta_hat[j]*X_j)
                    self.beta_hat[j] = self.__soft_threshold(__rio_j, self.penalty)
            residual = self.__Y - self.__X.dot(self.beta_hat)
            mse = (1 / self.__N) * (residual.T.dot(residual) + self.penalty * np.sum(np.abs(self.beta_hat)))
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