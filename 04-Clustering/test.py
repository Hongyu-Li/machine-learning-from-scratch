from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

centers = [[1, 1], [-1, -1], [1, -1]]
n_samples=750
X, y = make_blobs(n_samples=n_samples, centers= centers, cluster_std=0.4,
                  random_state =0)

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()