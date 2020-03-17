import numpy as np
from utils import compute_distance
from copy import deepcopy

class LVQ:
    def __init__(self, K=5, max_iter=500, eta=0.1):
        self.K = K
        self.max_iter = max_iter
        self.centroids = None
        self.eta = eta

    def __find_closest_cluster(self, data):
        dist = [compute_distance(data, c) for c in self.centroids]
        return np.argmin(dist)

    def fit(self, X, y, prior_labels):
        n, p = X.shape
        self.centroids = np.zeros((self.K, p))
        for _ in range(self.max_iter):
            x_idx = np.random.choice(range(n), 1)
            p_idx = self.__find_closest_cluster(X[x_idx])
            temp_p = deepcopy(self.centroids[p_idx])
            if y[x_idx] == prior_labels[p_idx]:
                self.centroids[p_idx] += self.eta * (X[x_idx] - self.centroids[p_idx]).flatten()
            else:
                self.centroids[p_idx] -= self.eta * (X[x_idx] - self.centroids[p_idx]).flatten()
            if not (self.centroids[p_idx] - temp_p).any():
                break

    def predict(self, X_new):
        return self.__find_closest_cluster(X_new)




