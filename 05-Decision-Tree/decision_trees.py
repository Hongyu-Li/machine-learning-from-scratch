import numpy as np
from utils import calculate_cross_entropy, calculate_gini, calculate_square_loss


# '''
# Useful Notes: https://levelup.gitconnected.com/building-a-decision-tree-from-scratch-in-python-machine-learning-from-scratch-part-ii-6e2e56265b19
# Here, we implement only binary decision trees.
# '''


class TreeStump:
    def __init__(self, X, y, features, samples_id, depth, max_depth):
        self.X = X
        self.y = y
        self.features = features
        self.samples_count = len(X)
        self.samples = samples_id
        self.is_leaf = False
        self.depth = depth
        self.max_depth = max_depth
        self.get_var_split()

    def get_var_split(self):
        pass

    def split_dataset_by_splitting_feature(self):
        left_idx = self.X[:, self.split_feature] < self.split_val
        right_idx = self.X[:, self.split_feature] >= self.split_val
        left_X, right_X = self.X[left_idx], self.X[right_idx]
        left_y, right_y = self.y[left_idx], self.y[right_idx]
        return left_X, left_idx, left_y, right_X, right_idx, right_y

    def find_best_split_feature_and_value(self):
        features_loss = []
        split_candidates = []
        for feature in self.features:
            split_val, loss = self.get_best_split_value(self.X[:, feature])
            features_loss.append(loss)
            split_candidates.append(split_val)
        split_feature = np.argmin(features_loss)
        return split_candidates[split_feature], self.features[split_feature]

    def get_best_split_value(self, feature):
        candidates = np.unique(feature)
        loss = []
        for i in candidates:
            left = self.y[feature < i]
            right = self.y[feature >= i]
            total_loss = self.calculate_loss(left, right)
            loss.append(total_loss)
        best_idx = np.argmin(loss)
        return candidates[best_idx], loss[best_idx]

    def calculate_loss(self, *args):
        return 0

    def fit(self):
        return [self.predict(i) for i in self.X]

    def predict(self, X_new):
        return np.mean(self.y)


class RegressionTree(TreeStump):
    def __init__(self, X, y, features, samples_id, depth, max_depth):
        super().__init__(X, y, features, samples_id, depth, max_depth)

    def __should_continue(self):
        return self.samples_count > 5 and len(self.features) != 0 and self.depth < self.max_depth - 1

    def get_var_split(self):
        if self.__should_continue():
            self.split_val, self.split_feature = super().find_best_split_feature_and_value()
            left_X, left_idx, left_y, right_X, right_idx, right_y = super().split_dataset_by_splitting_feature()
            # self.features.pop(split_feature)
            self.left = RegressionTree(left_X, left_y, list(self.features), self.samples[left_idx], self.depth + 1,
                                       self.max_depth)
            self.right = RegressionTree(right_X, right_y, list(self.features), self.samples[right_idx], self.depth + 1,
                                        self.max_depth)
        else:
            self.is_leaf = True

    def calculate_loss(self, *args):
        loss = 0
        for i in args:
            if len(i) != 0:
                loss += calculate_square_loss(i)
        return loss

    def predict(self, X_new):
        if self.is_leaf: return np.mean(self.y)
        node = self.left if X_new[self.split_feature] < self.split_val else self.right
        return node.predict(X_new)


class ClassificationTree(TreeStump):
    def __init__(self, X, y, features, samples_id, depth, max_depth, metric='entropy'):
        self.metric = metric
        super().__init__(X, y, features, samples_id, depth, max_depth)

    def __should_continue(self):
        return self.samples_count != 0 and len(self.features) != 0 and len(np.unique(
                self.y)) != 1 and self.depth < self.max_depth - 1

    def get_var_split(self):
        if self.__should_continue():
            self.split_val, self.split_feature = super().find_best_split_feature_and_value()
            left_X, left_idx, left_y, right_X, right_idx, right_y = super().split_dataset_by_splitting_feature()
            # self.features.pop(split_feature)
            self.left = ClassificationTree(left_X, left_y, list(self.features), self.samples[left_idx], self.depth + 1,
                                           self.max_depth, self.metric)
            self.right = ClassificationTree(right_X, right_y, list(self.features), self.samples[right_idx],
                                            self.depth + 1, self.max_depth, self.metric)
        else:
            self.is_leaf = True

    def calculate_loss(self, *args):
        loss = 0
        for i in args:
            if len(i) != 0:
                weight = len(i) / len(self.y)
                if self.metric == 'entropy':
                    loss += weight * calculate_cross_entropy(i)
                elif self.metric == 'gini':
                    loss += weight * calculate_gini(i)
            else:
                return float('Inf')
        if self.metric == 'entropy':
            loss = calculate_cross_entropy(self.y) - loss
        return loss

    def __majority_vote(self):
        count = []
        y_val = np.unique(self.y)
        for i in y_val:
            count.append(np.sum(self.y == i))
        return y_val[np.argmax(count)]

    def predict(self, X_new):
        if self.is_leaf: return self.__majority_vote()
        node = self.left if X_new[self.split_feature] < self.split_val else self.right
        return node.predict(X_new)


class DecisionTrees:
    def __init__(self, max_depth=5, classification=True, metric='entropy'):
        self.reg = None
        self.classification = classification
        self.metric = metric
        self.max_depth = max_depth

    def fit(self, X, y):
        n, p = X.shape
        features = [i for i in range(p)]
        samples = np.asarray([i for i in range(n)])
        if self.classification:
            self.reg = ClassificationTree(X, y, features, samples, 0, self.max_depth, self.metric)
        else:
            self.reg = RegressionTree(X, y, features, samples, 0, self.max_depth)
        return self.reg.fit()

    def predict(self, X_new):
        return self.reg.predict(X_new)
