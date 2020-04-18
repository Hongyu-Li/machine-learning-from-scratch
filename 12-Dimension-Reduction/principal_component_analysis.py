import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from kernels import linear_kernel, rbf_kernel, polynomial_kernel


class PCA:
    """
    SVD for PCA
    """
    def __init__(self):
        self.X = None
        self.W = None

    def fit(self, X, d):
        self.X = X - np.mean(X, 0)
        U, S, V = np.linalg.svd(self.X)
        # S is the eigenvalues, V is the eigenvectors
        indices = np.argsort(S)[::-1][:d]
        self.W = V[:, indices]
        return self.X.dot(self.W)

    def predict(self, X_new):
        centered_X_new = X_new - np.mean(self.X, 0)
        return centered_X_new.dot(self.W)


class KPCA:
    def __init__(self, kernel=linear_kernel()):
        self.X = None
        self.W = None
        self.kernel = kernel
        self.n = 0

    def fit(self, X, d):
        self.X = X - np.mean(X, 0)
        self.n = X.shape[0]
        kernel_mat = self.__get_kernel_matrix()
        eigenvalues, eigenvectors = np.linalg.eigh(kernel_mat)
        indices = np.argsort(eigenvalues)[::-1][:d]
        self.W = eigenvectors[:, indices]
        return kernel_mat.dot(self.W)

    def __get_kernel_matrix(self):
        kernel_mat = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                kernel_mat[i, j] = self.kernel(self.X[i], self.X[j])
        return kernel_mat

    def predict(self, X_new):
        centered_X_new = X_new - np.mean(self.X, 0)
        return self.kernel(centered_X_new, self.X).dot(self.W)


if __name__ == '__main__':
    X, color = datasets.make_swiss_roll(n_samples=1500)
    # X, color = datasets.make_s_curve(1500, random_state=0)
    # pca = PCA()
    pca = KPCA()
    kpca = KPCA(kernel=rbf_kernel(0.01))
    X_pca = pca.fit(X, 2)
    X_kpca = kpca.fit(X,2)
    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.set_title("Original data")
    ax = fig.add_subplot(222)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title('Projected data-PCA')
    ax = fig.add_subplot(223)
    ax.scatter(X_kpca[:, 0], X_kpca[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title('Projected data-KPCA')
    plt.show()
