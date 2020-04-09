import numpy as np


def linear_kernel(**args):
    def f(x1, x2):
        return np.inner(x1, x2)

    return f


def rbf_kernel(gamma, **args):
    # a version of gaussian kernel
    def f(x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)
    return f


def polynomial_kernel(d, **args):
    def f(x1, x2):
        return np.inner(x1, x2) ** d

    return f
