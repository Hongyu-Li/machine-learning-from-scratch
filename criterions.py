from utils import calculate_square_loss
import numpy as np


def calculate_rss(y, preds):
    return np.sum((y - preds) ** 2)


def calculate_sst(y):
    return np.sum((y - np.mean(y)) ** 2)


def calculate_r2(y, preds):
    rss = calculate_rss(y, preds)
    sst = calculate_sst(y)
    return 1 - rss / sst


def calculate_adjust_r2(y, preds, k):
    n = len(y)
    numerator = calculate_rss(y, preds) / (n - k - 1)
    denominator = calculate_sst(y) / (n - 1)
    return 1 - numerator / denominator


def calculate_aic(y, preds, k):
    rss = calculate_rss(y, preds)
    return 2 * k - 2 * np.log(rss)


def calculate_bic(y, preds, k):
    n = len(y)
    rss = calculate_rss(y, preds)
    return 2 * np.log(n) * k - 2 * np.log(rss)