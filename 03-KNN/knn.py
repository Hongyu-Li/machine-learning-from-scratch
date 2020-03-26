import numpy as np
import matplotlib.pyplot as plt

from utils import confusion_matrix,compute_distance


class KNN:
    """
    This KNN is a vanilla version which means the distance is euclidean and the weights strategy is uniform
    """
    def __init__(self, K=3, classification=True):
        self.K = K
        self.classification = classification

    def __get_neighbors(self, X_new, X):
        distance = [compute_distance(X_new, data) for data in X]
        return np.argsort(distance)[:self.K]

    def predict(self, X_new, X, y):
        neighs = self.__get_neighbors(X_new, X)
        if self.classification:
            counts = np.bincount(y[neighs])
            return np.argmax(counts)
        else:
            return np.mean(y[neighs])

    def __fit(self, X, y):
        pred = []
        for i in range(len(X)):
            X_truncated = np.vstack((X[:i], X[(i + 1):]))
            y_truncated = np.concatenate((y[:i], y[(i + 1):]))
            pred.append(self.predict(X[i], X_truncated, y_truncated))
        return pred

    def evaluate(self, X, y, with_plot=False):
        pred = self.__fit(X, y)
        if self.classification:
            return confusion_matrix(y, pred)
        else:
            if with_plot:
                plt.figure(figsize=(12, 8))
                plt.scatter(y, pred)
                plt.xlabel('y')
                plt.ylabel('y_pred')
            return np.mean((y - pred) ** 2)

    def plot_cv_k(self, k_lst, X, y):
        errors = []
        for k in k_lst:
            self.K = k
            if self.classification:
                err = 1 - np.sum(np.diag(self.evaluate(X, y))) / np.sum(self.evaluate(X, y))
            else:
                err = self.evaluate(X, y)
            errors.append(err)

        plt.figure(figsize=(12, 8))
        plt.plot(k_lst, errors)
        plt.xlabel('k')
        plt.ylabel('errors')




