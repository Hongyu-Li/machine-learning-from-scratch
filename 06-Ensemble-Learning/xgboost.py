import numpy as np
import sys
from copy import deepcopy
from activations import Sigmoid
import pandas as pd
from losses import MSE, LogLoss
from decision_trees import TreeStump


class XGBoostRegressionTree(TreeStump):
    '''
    The algorithm for splitting point without using weighted quantile sketch
    '''

    def __init__(self, X, y, features, samples_id, depth, max_depth, lambdap=0.01, gamma=0.01, loss='mse'):
        self.lambdap = lambdap
        self.gamma = gamma
        self.loss_metric = loss
        if loss == 'mse':
            self.loss = MSE()
        elif loss == 'log':
            self.loss = LogLoss()
        super().__init__(X, y, features, samples_id, depth, max_depth)

    def __should_continue(self):
        return self.samples_count > 5 and len(self.features) != 0 and self.depth < self.max_depth - 1

    def get_var_split(self):
        if self.__should_continue():
            self.split_val, self.split_feature = super().find_best_split_feature_and_value()
            left_X, left_idx, left_y, right_X, right_idx, right_y = self.split_dataset_by_splitting_feature()
            self.left = XGBoostRegressionTree(left_X, left_y, list(self.features), self.samples[left_idx], self.depth + 1,
                                       self.max_depth, self.lambdap, self.gamma, self.loss_metric)
            self.right = XGBoostRegressionTree(right_X, right_y, list(self.features), self.samples[right_idx], self.depth + 1,
                                        self.max_depth, self.lambdap, self.gamma, self.loss_metric)
        else:
            self.is_leaf = True

    def split_dataset_by_splitting_feature(self):
        left_idx = self.X[:, self.split_feature] < self.split_val
        right_idx = self.X[:, self.split_feature] >= self.split_val
        left_X, right_X = self.X[left_idx], self.X[right_idx]
        y, y_pred = self.__split_y_and_pred(self.y)
        left_y = np.concatenate((y[left_idx], y_pred[left_idx]))
        right_y = np.concatenate((y[right_idx], y_pred[right_idx]))
        return left_X, left_idx, left_y, right_X, right_idx, right_y

    def __split_y_and_pred(self, y):
        mid_idx = len(y) // 2
        return y[:mid_idx], y[mid_idx:]

    def get_best_split_value(self, feature):
        candidates = np.unique(feature)
        loss = []
        for i in candidates:
            y, y_pred = self.__split_y_and_pred(self.y)
            left = np.concatenate((y[feature < i], y_pred[feature < i]))
            right = np.concatenate((y[feature >= i], y_pred[feature >= i]))
            total_loss = self.calculate_loss(left, right)
            loss.append(total_loss)
        best_idx = np.argmin(loss)
        return candidates[best_idx], loss[best_idx]

    def calculate_loss(self, *args):
        gradients = []
        hessians = []
        for group in args:
            y, y_pred = self.__split_y_and_pred(group)
            gradients.append(self.loss.gradient(y, y_pred).sum())
            hessians.append(self.loss.hessian(y,y_pred).sum())
        after_gain = np.sum(np.power(gradients,2) / np.add(hessians,self.lambdap))
        before_gain = np.sum(gradients)**2 / (np.sum(hessians) + self.lambdap)
        gain = after_gain - before_gain - self.gamma
        return -gain

    def predict(self, X_new):
        if self.is_leaf: return self.__calculate_leaf_score()
        node = self.left if X_new[self.split_feature] < self.split_val else self.right
        return node.predict(X_new)

    def __calculate_leaf_score(self):
        y, y_pred = self.__split_y_and_pred(self.y)
        numerator = np.sum(self.loss.gradient(y, y_pred))
        denominator = np.sum(self.loss.hessian(y, y_pred)) + self.lambdap
        return - numerator / denominator


# for binary classification and regression
class XGBoost:
    def __init__(self, n_trees=100, classification=True, learning_rate=0.01, max_depth=10):
        self.n_trees = n_trees
        self.classification = classification
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.X = None
        self.y = None
        self.weak_learners = []
        self.preds = []
        self.Sigmoid = Sigmoid()

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n, self.p = X.shape
        self.__initialize_base_learner()
        self.__build_weak_learners()
        if self.classification:
            labels = deepcopy(self.Sigmoid(self.preds))
            labels[labels > 0.5] = 1
            labels[labels <= 0.5] = 0
            return labels
        else:
            return self.preds

    def predict(self, X_new):
        y_pred = 0
        for i in range(self.n_trees):
            tree = self.weak_learners[i]
            y_pred += np.multiply(self.learning_rate, tree.predict(X_new))

        if self.classification:
            y_pred = self.Sigmoid(y_pred)
            return 1 if y_pred > 0.5 else 0
        else:
            return y_pred

    def __build_weak_learners(self):
        features = np.asarray([i for i in range(self.p)])
        samples_id = np.asarray([i for i in range(self.n)])
        for _ in range(self.n_trees):
            y = np.concatenate((self.y, self.preds))
            if self.classification:
                tree = XGBoostRegressionTree(self.X, y, features, samples_id, 0, self.max_depth, loss='log')
            else:
                tree = XGBoostRegressionTree(self.X, y, features, samples_id, 0, self.max_depth, loss='mse')
            self.preds += np.multiply(self.learning_rate, tree.fit())
            self.weak_learners.append(tree)

    def __initialize_base_learner(self):
        self.preds = np.zeros(self.y.shape)


if __name__ == '__main__':
    data = pd.read_csv('../data/diabetes.csv', sep=',')
    X = np.asarray(data.iloc[:, :-1])
    y = np.asarray(data.iloc[:, -1])
    model = XGBoost(classification=True,n_trees=200)
    preds = model.fit(X,y)