import numpy as np

from activations import Sigmoid


class LogisticRegression:
    def __init__(self, n_iter=1000, learning_rate=1e-4, intercept=False):
        self.weights = None
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.intercept = intercept
        self.sigmoid = Sigmoid()

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float128)
        Y = np.asarray(Y, dtype=np.float128)
        p = X.shape[1]
        N = X.shape[0]

        if self.intercept:
            ones = np.reshape(np.ones(N), (N, 1))
            X = np.hstack((ones, X))
            p = p + 1
        self.weights = np.zeros(p)

        training_loss = []
        self.weights = np.random.uniform(-1/N, 1/N, size=p)
        for i in range(self.n_iter):
            Y_pred = self.sigmoid(X.dot(self.weights))
            # use mean to avoid overflow issue
            loss = np.mean(-Y * np.log(Y_pred) - (1 - Y) * np.log(1 - Y_pred))
            training_loss.append(loss)
            self.weights -= self.learning_rate * (X.T.dot(Y_pred-Y) / N)
        return training_loss

    def predict(self,X_new):
        return X_new.dot(self.weights)
