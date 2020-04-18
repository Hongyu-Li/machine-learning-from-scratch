import numpy as np
from utils import compute_distance
from sklearn import datasets
import matplotlib.pyplot as plt


class MDS:
    def __init__(self):
        self.B = None
        self.W = None
        self.n = 0

    def fit(self, X, d):
        self.n = X.shape[0]
        distances = self.__get_samples_distance_matrix()
        self.__compute_inner_product_after_reduction(distances)
        eigenvalues, eigenvectors = np.linalg.eigh(self.B)
        indices = np.argsort(eigenvalues)[::-1][:d]
        self.W = eigenvectors[:, indices]
        return self.W.dot(np.diag(eigenvalues[indices]) ** 0.5)

    def __compute_inner_product_after_reduction(self, distances):
        self.B = np.zeros((self.n, self.n))
        dist_mean = np.sum(distances) / (self.n ** 2)
        for i in range(self.n):
            for j in range(self.n):
                dist_i = np.mean(distances[i])
                dist_j = np.mean(distances[:, j])
                self.B[i, j] = -0.5 * (distances[i, j] ** 2 - dist_i ** 2 - dist_j ** 2 + dist_mean ** 2)

    def __get_samples_distance_matrix(self):
        distances = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                distances[i, j] = compute_distance(self.X[i], self.X[j])
        return distances


if __name__ == '__main__':
    X, color = datasets.make_swiss_roll(n_samples=1500)
    mds = MDS()
    X_r = mds.fit(X, 2)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.set_title("Original data")
    ax = fig.add_subplot(122)
    ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title('MDS')
    plt.show()
