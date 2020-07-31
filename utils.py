import numpy as np


def compute_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def confusion_matrix(y, y_pred):
    classes = np.unique(y)
    pred_classes = np.unique(y_pred)
    mat = np.zeros((len(classes), len(pred_classes)))
    for k in range(len(classes)):
        for j in range(len(pred_classes)):
            mat[k, j] = np.sum((y == classes[k]) * (y_pred == pred_classes[j]))
    return mat


def calculate_cross_entropy(y):
    entropy = lambda x: np.mean(y == x) * np.log2(np.mean(y == x))
    return np.asarray([entropy(i) for i in np.unique(y)]).sum()


def calculate_gini(y):
    # add a small value 1e-10 to avoid underflow issue of np.log2
    gini = lambda x: np.mean(y == x) * np.log2(1 - np.mean(y == x) + 1e-10)
    return np.asarray([gini(i) for i in np.unique(y)]).sum()


def calculate_square_loss(i):
    return np.sum((i - np.mean(i)) ** 2)
