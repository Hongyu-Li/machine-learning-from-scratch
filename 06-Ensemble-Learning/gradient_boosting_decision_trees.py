import numpy as np
from activations import Sigmoid
import sys
from copy import deepcopy
import pandas as pd

sys.path.insert(0, '../05-Decision-Tree/')
from decision_trees import RegressionTree



# for binary classification and regression
class GBDT:
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
        if self.classification:
            y_pred = np.log2(np.sum(self.y) / np.sum(1 - self.y))
        else:
            y_pred = np.mean(self.y)

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
            if self.classification:
                negative_grad = self.y - self.preds
            else:
                negative_grad = self.y - self.Sigmoid(self.preds)
            tree = RegressionTree(self.X, negative_grad, features, samples_id, 0, self.max_depth)
            self.preds += np.multiply(self.learning_rate, tree.fit())
            self.weak_learners.append(tree)

    def __initialize_base_learner(self):
        if self.classification:
            self.preds = np.asarray([np.log2(np.sum(self.y) / np.sum(1 - self.y))] * self.n)
        else:
            self.preds = np.asarray([np.mean(self.y)] * self.n)


# if __name__ == '__main__':
#     data = pd.read_csv('../data/diabetes.csv', sep=',')
#     X = np.asarray(data.iloc[:, :-1])
#     y = np.asarray(data.iloc[:, -1])
#     model = GBDT(classification=True, n_trees=50, learning_rate=1, max_depth=20)
#     preds = model.fit(X, y)