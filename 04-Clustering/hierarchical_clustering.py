import numpy as np
from utils import compute_distance


class Hcluster:
    def __init__(self, K=3, distance='avg'):
        self.K = K
        self.distance_metric = distance
        self.clusters = []
        self.distances = []

    def __get_cluster_distance(self, a, b):
        temp = np.vstack((a, b))
        if self.distance_metric == 'min':
            return np.min(temp, axis=0)
        elif self.distance_metric == 'max':
            return np.max(temp, axis=0)
        elif self.distance_metric == 'avg':
            return np.mean(temp, axis=0)
        else:
            raise Exception('Distance metric is not valid.')

    def __update_distance(self, x_pos, y_pos):
        mat_1 = self.distances[x_pos]
        mat_2 = self.distances[y_pos]
        candidate = self.__get_cluster_distance(mat_1,mat_2)
        self.distances[x_pos] = candidate
        self.distances[:,x_pos] = self.distances[x_pos]
        self.distances = np.vstack((self.distances[:y_pos], self.distances[(y_pos+1):]))
        self.distances = np.hstack((self.distances[:,:y_pos], self.distances[:,(y_pos+1):]))

    def __find_closest_cluster(self):
        classes = len(self.clusters)
        min_dist = float('Inf')
        x_pos, y_pos = 0, 0
        for i in range(classes):
            for j in range(i+1, classes):
                curr_dist = self.distances[i, j]
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    x_pos, y_pos = i, j
        return x_pos, y_pos


    def fit(self, X):
        n = X.shape[0]
        self.clusters =[[i] for i in range(n)]
        self.distances = np.full((n, n), float('Inf'))
        for i in range(n-1):
            for j in range(i+1, n):
                self.distances[i, j] = compute_distance(X[i], X[j])
                self.distances[j, i] = self.distances[i,j]

        while len(self.clusters) > self.K:
            x_pos, y_pos = self.__find_closest_cluster()
            self.__update_distance(x_pos, y_pos)
            self.clusters[x_pos] += self.clusters[y_pos]
            self.clusters.pop(y_pos)

    def get_labels(self, n):
        labels = [0]*n
        for i in range(len(self.clusters)):
            for j in self.clusters[i]:
                labels[j] = i + 1
        return labels




