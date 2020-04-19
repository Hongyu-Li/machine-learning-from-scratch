import numpy as np
from utils import compute_distance
from sklearn import datasets
import matplotlib.pyplot as plt

class LLE:
    def __init__(self):
        self.W = None
        self.X = None
        self.n, self.p = 0,0

    def fit(self, X, k, d):
        self.X = X
        self.n, self.p = X.shape
        distances = self.__get_samples_distance_matrix()
        self.W = np.zeros((self.n, self.n))
        for i in range(self.n):
            indices = np.argsort(distances[i])[1:(k+1)]
            Z = X[indices] - X[i]
            C = Z.dot(Z.T)
            # Local covariance. Regularize because our data is
            # low-dimensional (K > D). dim(C) = [K, K]
            C += np.eye(k) * np.trace(C) * 0.001
            w = np.linalg.solve(C, np.ones(k))
            self.W[i, indices] = w / np.sum(w)
            M = (np.eye(self.n) - self.W).T.dot(np.eye(self.n) - self.W)
            eigenvalues, eigenvectors = np.linalg.eigh(M)
            return eigenvectors[:, np.argsort(eigenvalues)[1:(d+1)]]

    def __get_samples_distance_matrix(self):
        distances = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                distances[i, j] = compute_distance(self.X[i], self.X[j])
        return distances


if __name__ == '__main__':
    X, color = datasets.make_s_curve(1500, random_state=0)
    lle = LLE()
    X_r = lle.fit(X, 10, 2)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.set_title("Original data")
    ax = fig.add_subplot(122)
    ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title('LLE')
    plt.show()