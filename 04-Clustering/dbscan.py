import numpy as np
from utils import compute_distance


class DBSCAN:
    def __init__(self, eps=1, min_pts=10):
        self.eps = eps
        self.min_pts = min_pts
        self.visited_samples = []
        self.labels = []

    def __get_neighbors(self, X, idx):
        neighs = []
        for i in range(len(X)):
            d = compute_distance(X[idx], X[i])
            if d <= self.eps and i != idx:
                neighs.append(i)
        return np.asarray(neighs)

    def __expand_cluster(self, neighs, X, C):
        for j in neighs:
            if (j not in self.visited_samples) and self.labels[j] != -1:
                pts = self.__get_neighbors(X, j)
                self.visited_samples.append(j)
                if len(pts) >= self.min_pts:
                    self.labels[j] = C
                    # recursively expand cluster
                    self.__expand_cluster(pts, X, C)
                else:
                    # noisy point
                    self.labels[j] = -1

    def fit(self, X):
        n = X.shape[0]
        C = 0
        self.labels = np.full(n, 0)
        for i in range(n):
            if i in self.visited_samples or self.labels[i] == -1:
                continue
            else:
                neighs = self.__get_neighbors(X, i)
                self.visited_samples.append(i)
                if len(neighs) >= self.min_pts:
                    self.labels[neighs] = C
                    self.__expand_cluster(neighs, X, C)
                    C += 1
                else:
                    # noisy point
                    self.labels[i] = -1

