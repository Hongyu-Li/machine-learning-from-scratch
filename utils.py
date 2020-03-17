import numpy as np

def compute_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))