import numpy as np


class binaryLDA:
    def __init__(self):
        self.w = None

    def fit(self, X, Y):
        X_0 = X[Y == 0]
        X_1 = X[Y == 1]

        mu_0 = np.mean(X_0, axis=0)
        mu_1 = np.mean(X_1, axis=0)
        # in order to get the sum
        cov_0 = (len(X_0)-1) * np.cov(X_0.T)
        cov_1 = (len(X_1)-1) * np.cov(X_1.T)
        s_w = cov_0 + cov_1

        # svd to get inverse
        self.w = np.linalg.pinv(s_w).dot(mu_0 - mu_1)

    def predict(self, X_new):
        return self.w.T.dot(X_new)


class multiLDA:
    def __init__(self):
        self.w = None

    def fit(self, X, Y, d):
        n_classes = len(np.unique(Y))
        p = X.shape[1]
        mu = np.reshape(np.mean(X, axis=0), (p, 1))
        s_w = np.zeros((p, p))
        s_b = np.zeros((p, p))
        for i in range(n_classes):
            X_i = X[Y == i]
            mu_i = np.reshape(np.mean(X_i, axis=0), (p, 1))
            s_w += (len(X_i)-1) * np.cov(X_i.T)
            s_b += len(X_i) * (mu_i - mu).dot((mu_i - mu).T)

        # eigh used for symmetric matrix
        eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(s_w).dot(s_b))
        indices = np.argsort(eigenvalues)[::-1]
        self.w = eigenvectors[:, indices][:,:d]

    def predict(self, X_new):
        return self.w.T.dot(X_new)
