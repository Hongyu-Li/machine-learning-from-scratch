import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RegressionTree:
    def __init__(self, X, y, features, samples_id, depth, max_depth):
        self.X = X
        self.y = y
        self.features = features
        self.samples_count = len(X)
        self.samples = samples_id
        self.is_leaf = False
        self.depth = depth
        self.max_depth = max_depth
        self.__get_var_split()

    def __get_var_split(self):
        if self.samples_count > 5 and len(self.features) != 0 and self.depth < self.max_depth - 1:
            self.split_val, self.split_feature = self.__find_best_split_feature_and_value()
            left_idx = self.X[:, self.split_feature] < self.split_val
            right_idx = self.X[:, self.split_feature] >= self.split_val
            left_X, right_X = self.X[left_idx], self.X[right_idx]
            left_y, right_y = self.y[left_idx], self.y[right_idx]
            # self.features.pop(split_feature)
            self.left = RegressionTree(left_X, left_y, list(self.features), self.samples[left_idx], self.depth + 1,
                                       self.max_depth)
            self.right = RegressionTree(right_X, right_y, list(self.features), self.samples[right_idx], self.depth + 1,
                                        self.max_depth)
        else:
            self.is_leaf = True

    def __find_best_split_feature_and_value(self):
        features_loss = []
        split_candidates = []
        for feature in self.features:
            split_val, loss = self.__get_best_split_value(self.X[:, feature])
            features_loss.append(loss)
            split_candidates.append(split_val)
        split_feature = np.argmin(features_loss)
        return split_candidates[split_feature], self.features[split_feature]

    def __get_best_split_value(self, feature):
        candidates = np.unique(feature)
        loss = []
        for i in candidates:
            left = feature[feature < i]
            right = feature[feature >= i]
            total_loss = self.__calculate_loss(left, right)
            loss.append(total_loss)
        best_idx = np.argmin(loss)
        return candidates[best_idx], loss[best_idx]

    def __calculate_loss(self, *args):
        loss = 0
        for i in args:
            if len(i) != 0:
                loss += np.sum((i - np.mean(i)) ** 2)
        return loss

    def fit(self):
        return [self.predict(i) for i in self.X]

    def predict(self, X_new):
        if self.is_leaf: return np.mean(self.y)
        node = self.left if X_new[self.split_feature] < self.split_val else self.right
        return node.predict(X_new)


# if __name__ == '__main__':
#     data = pd.read_csv('../data/housing.csv', sep=',')
#     X = np.asarray(data.iloc[:, :-1])
#     Y = np.asarray(data.iloc[:, -1])
#     n, p = X.shape
#     features = [i for i in range(p)]
#     samples = np.asarray([i for i in range(n)])
#     reg = RegressionTree(X, Y, features, samples, 0, 5)


class DecisionTrees:
    def __init__(self):
        self.reg = None

    def fit(self, X, y):
        n, p = X.shape
        features = [i for i in range(p)]
        samples = np.asarray([i for i in range(n)])
        self.reg = RegressionTree(X, y, features, samples, 0, 5)
        return self.reg.fit()

    def predict(self, X_new):
        return self.reg.predict(X_new)



