import numpy as np


class Sigmoid:
    def __call__(self, x):
        # avoid overflow & underflow issue
        x = (x - np.min(x)) / (np.max(x) -np.min(x))
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class Linear:
    def __call__(self, x):
        return x

    def gradient(self, x):
        return 1


class Tanh:
    def __call__(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def gradient(self, x):
        return (1 - self.__call__(x)) ** 2


class Softmax:
    def __call__(self, x):
        # subtract the maximum to avoid overflow
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)
