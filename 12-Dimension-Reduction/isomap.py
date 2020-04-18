import numpy as np
from scipy.sparse.csgraph import dijkstra
from sklearn import datasets
import matplotlib.pyplot as plt
from utils import compute_distance


class Isomap:
    def __init__(self):
        self.B = None
        self.W = None
        self.n = 0
        self.X = None

    def fit(self, X, k, d):
        self.n = X.shape[0]
        self.X = X
        updated_distance = self.__get_dijkstra_updated_distances(k)
        self.__compute_inner_product_after_reduction(updated_distance)
        eigenvalues, eigenvectors = np.linalg.eigh(self.B)
        indices = np.argsort(eigenvalues)[::-1][:d]
        self.W = eigenvectors[:, indices]
        return self.W.dot(np.diag(eigenvalues[indices]) ** 0.5)

    def __get_dijkstra_updated_distances(self, k):
        distances = self.__get_samples_distance_matrix()
        for i in range(self.n):
            distances[i, np.argsort(distances[i])[k:]] = float('Inf')
        updated_distance = dijkstra(distances, return_predecessors=False)
        return updated_distance

    def __compute_inner_product_after_reduction(self, distances):
        self.B = np.zeros((self.n, self.n))
        dist_mean = np.mean(distances[distances != float('Inf')])
        for i in range(self.n):
            for j in range(self.n):
                dist_i = np.mean(distances[i][distances[i]!=float('Inf')])
                dist_j = np.mean(distances[:,j][distances[:,j]!=float('Inf')])
                self.B[i, j] = -0.5 * (distances[i, j] ** 2 - dist_i ** 2 - dist_j ** 2 + dist_mean ** 2)

    def __get_samples_distance_matrix(self):
        distances = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                distances[i, j] = compute_distance(self.X[i], self.X[j])
        return distances

if __name__ == '__main__':
    X, color = datasets.make_s_curve(1500, random_state=0)
    isomap = Isomap()
    X_r = isomap.fit(X, 10, 2)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.set_title("Original data")
    ax = fig.add_subplot(122)
    ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title('ISOMAP')
    plt.show()