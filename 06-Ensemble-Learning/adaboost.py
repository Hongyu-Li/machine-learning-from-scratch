import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Decision stump used as weak classifier
from utils import confusion_matrix


class DecisionStump:
    def __init__(self, split_feature_idx, split_val):
        self.split_feature_idx = split_feature_idx
        self.split_val = split_val

    def fit(self, X):
        return [self.predict(i) for i in X]

    def predict(self, X_new):
        return -1 if X_new[self.split_feature_idx] < self.split_val else 1


class Adaboost:
    def __init__(self, n_trees=100):
        self.n_trees = n_trees
        self.X = None
        self.y = None
        self.n, self.p = 0, 0
        self.weights = []
        self.alpha = []
        self.weak_learners = []

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n, self.p = X.shape
        self.weights = np.asarray([1 / self.n] * self.n)
        self.__train_weak_learners()
        pred_mat = np.zeros((self.n, self.n_trees))
        for i in range(self.n_trees):
            classifier = self.weak_learners[i]
            pred_mat[:,i] = self.alpha[i] * classifier.fit(self.X)
        return self.__majority_vote(np.sum(pred_mat, 1))

    def predict(self, X_new):
        preds = np.zeros(self.n_trees)
        for i in range(self.n_trees):
            classifier = self.weak_learners[i]
            preds[i] = self.alpha[i] * classifier.predict(X_new)
        return self.__majority_vote(preds)

    def __majority_vote(self, preds):
        preds[preds > 0] = 1
        preds[preds < 0] = -1
        return preds

    def __train_weak_learners(self):
        for _ in range(self.n_trees):
            split_val, split_feature_idx, min_error = self.__find_best_split_feature(range(self.p))
            if min_error < 0.5:
                stump = DecisionStump(split_feature_idx, split_val)
                model_weight = 0.5 * np.log2(((1 - min_error) / min_error) + 1e-10)
                self.weak_learners.append(stump)
                self.alpha.append(model_weight)
                self.__update_sample_weights(stump, model_weight)

    def __update_sample_weights(self, stump, alpha):
        y_preds = stump.fit(self.X)
        self.weights *= np.exp(alpha * (y_preds != self.y))
        self.weights = self.weights / np.sum(self.weights)

    def __find_best_split_feature(self, features):
        features_err = []
        split_candidates = []
        for feature in features:
            split_val, err = self.__get_best_split_value(self.X[:, feature])
            features_err.append(err)
            split_candidates.append(split_val)
        split_feature = np.argmin(features_err)
        return split_candidates[split_feature], features[split_feature], np.min(features_err)

    def __get_best_split_value(self, feature):
        candidates = np.unique(feature)
        err = []
        for i in candidates:
            preds = np.zeros(self.n)
            preds[feature < i] = -1
            preds[feature >= i] = 1
            total_err = self.__calculate_error_rate(preds)
            err.append(total_err)
        best_idx = np.argmin(err)
        return candidates[best_idx], err[best_idx]

    def __calculate_error_rate(self, preds):
        return np.sum(self.weights * (preds == self.y)) / np.sum(self.weights)


if __name__ == '__main__':
    data = pd.read_csv('../data/diabetes.csv', sep=',')
    X = np.asarray(data.iloc[:, :-1])
    y = np.asarray(data.iloc[:, -1])
    y[y == 0] = -1
    model = Adaboost()
    preds = model.fit(X, y)
    mat = confusion_matrix(preds, y)
    print(mat)
