import numpy as np

def __calculate_loss(*args):
    loss = 0
    for i in args:
        if len(i) != 0:
            loss += np.sum((i - np.mean(i))**2)
    return loss

    def __get_best_split_value(feature):
        candidates = np.unique(feature)
        loss = []
        for i in candidates:
            left = feature[feature < i]
            right = feature[feature >= i]
            total_loss = __calculate_loss(left, right)
            loss.append(total_loss)
        best_idx = np.argmin(loss)
        return candidates[best_idx], loss[best_idx]