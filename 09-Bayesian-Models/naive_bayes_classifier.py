import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from utils import confusion_matrix


class NaiveBayes:
    """
    Naive Bayes Classifier with gaussian assumption
    """

    def __init__(self):
        self.parameters = []
        self.n, self.p = 0, 0
        self.classes = []
        self.X, self.y = None, None

    def fit(self, X, y):
        self.n, self.p = X.shape
        self.classes = np.unique(y)
        self.X = X
        self.y = y
        for i, c in enumerate(self.classes):
            samples_X = self.X[self.y == c]
            mu = np.mean(samples_X, 0)
            std = np.std(samples_X, 0)
            self.parameters.append([])
            for j in range(self.p):
                self.parameters[i].append({'mean': mu[j], 'std': std[j]})

    def predict(self, X_new):
        posteriors = []
        for i, c in enumerate(self.classes):
            likelihood = self.__calculate_likelihood(self.parameters[i], X_new)
            prior = np.sum(self.y == c) / self.n
            posteriors.append(prior * likelihood)
        return np.argmax(posteriors)

    def __calculate_likelihood(self, params, X):
        likelihood = 1
        for i, x in enumerate(X):
            likelihood *= (1 / np.sqrt(2 * np.pi * params[i]['std'])) * np.exp(
                        -(x - params[i]['mean']) ** 2) / (2 * (params[i]['std'] ** 2))
        return likelihood


# if __name__ == '__main__':
#     centers = [[1, 1], [-1, -1], [1, -1]]
#     n_samples = 750
#     X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.4,
#                       random_state=0)
#     model = NaiveBayes()
#     model.fit(X, y)
#     preds = [model.predict(x) for x in X]
#     mat = confusion_matrix(y, preds)
#     print(mat)
