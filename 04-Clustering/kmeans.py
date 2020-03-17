import numpy as np
from utils import compute_distance
from copy import deepcopy


class Kmeans:
    def __init__(self, K=3, max_iter=500):
        self.K = K
        self.max_iter = max_iter
        self.centroids = None

    def __get_centroids(self, X, labels):
        for k in range(self.K):
            group = X[labels == k]
            self.centroids[k] = np.mean(group, axis=0)

    def fit(self, X):
        n, p = X.shape
        c_label = np.random.randint(self.K, size=n)
        self.centroids = np.zeros((self.K, p))
        for _ in range(self.max_iter):
            prev_centroids = deepcopy(self.centroids)
            self.__get_centroids(X, c_label)
            for i in range(n):
                c_label[i] = self.predict(X[i])
            if not (prev_centroids - self.centroids).any():
                break
        return c_label

    def predict(self, X_new):
        return np.argmin([compute_distance(X_new, center) for center in self.centroids])

