import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, K=3, n_iter=500):
        self.K = K
        self.n_iter = n_iter
        self.n = 0
        self.p = 0
        self.alpha = None
        self.mu = None
        self.sigma = None
        self.expectations = None

    def __get_likelihoods(self, X):
        density = np.zeros((self.n, self.K))
        for j in range(self.K):
            density[:, j] = multivariate_normal.pdf(X, self.mu[j], self.sigma[j])
        return density

    def __expectation(self, post_prob):
        numerator = post_prob * np.repeat(self.alpha.T, self.n, axis=0)
        denominator = np.repeat(np.sum(numerator, 1)[:, np.newaxis], self.K, axis=1)
        self.expectations = numerator / denominator

    def __maximization(self, X):
        for k in range(self.K):
            weights = self.expectations[:, k].reshape((1, self.n))
            self.mu[k] = weights.dot(X) / np.sum(self.expectations, axis=0)[k]
            temp_cov = (X - self.mu[k]).T.dot((X - self.mu[k]) * weights.reshape(self.n, 1))
            self.sigma[k] = temp_cov / np.sum(self.expectations, axis=0)[k]
            self.alpha[k] = np.sum(weights) / self.n

    def fit(self, X):
        self.n, self.p = X.shape
        self.alpha = np.full((self.K, 1), 1 / self.K)
        self.mu = np.random.uniform(0, 1, self.K * self.p).reshape(self.K, self.p)
        self.sigma = np.repeat(np.diag(np.random.uniform(0, 1, self.p))[np.newaxis, :, :], self.K, axis=0)
        for _ in range(self.n_iter):
            post_prob = self.__get_likelihoods(X)
            self.__expectation(post_prob)
            self.__maximization(X)

        labels = np.argmax(self.expectations, 1)
        return labels
