import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


class PCA:
    def __init__(self):
        self.X = None
        self.W = None

    def fit(self, X, d):
        centered_X = X - np.mean(X, 0)
        U, S, V = np.linalg.svd(centered_X)
        # S is the eigenvalues, V is the eigenvectors
        indices = np.argsort(S)[::-1][:d]
        self.W = V[:, indices]
        return centered_X.dot(self.W)


if __name__ == '__main__':
    X, color = datasets.make_swiss_roll(n_samples=1500)
    model = PCA()
    X_r = model.fit(X, 2)
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.set_title("Original data")
    ax = fig.add_subplot(212)
    ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])
    plt.title('Projected data')
    plt.show()
