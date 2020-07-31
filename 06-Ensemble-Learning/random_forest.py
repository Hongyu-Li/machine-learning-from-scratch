import numpy as np
from decision_trees import ClassificationTree, RegressionTree
import pandas as pd
from utils import confusion_matrix


# RandomForest for binary classification and regression
class RandomForest:
    def __init__(self, n_trees=100, classification=True, max_depth=10, split_metric='gini'):
        self.n_trees = n_trees
        self.X = None
        self.y = None
        self.n, self.p = 0, 0
        self.max_depth = max_depth
        self.split_metric = split_metric
        self.classification = classification
        self.weak_learners = []
        np.random.seed(1)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n, self.p = X.shape
        self.__build_weak_learners()
        pred_mat = np.zeros((self.n, self.n_trees))
        for i in range(self.n_trees):
            classifier = self.weak_learners[i]
            pred_mat[:, i] = classifier.fit()
        if self.classification:
            return self.__majority_vote(pred_mat)
        else:
            return self.__get_average(pred_mat)

    def predict(self, X_new):
        preds = np.zeros(self.n_trees)
        for i in range(self.n_trees):
            classifier = self.weak_learners[i]
            preds[i] = classifier.predict(X_new)
        if self.classification:
            return self.__majority_vote(np.expand_dims(preds, 0))
        else:
            return self.__get_average(np.expand_dims(preds, 0))

    def __majority_vote(self, preds):
        y_preds = []
        for i in range(preds.shape[0]):
            count = []
            y_val = np.unique(preds[i])
            for j in y_val:
                count.append(np.sum(preds[i] == j))
            y_preds.append(y_val[np.argmax(count)])
        return y_preds

    def __get_average(self, preds):
        return np.mean(preds, 1)

    def __build_weak_learners(self):
        for _ in range(self.n_trees):
            bootstrap_feature = np.random.choice(range(self.p), self.p, replace=True)
            bootstrap_data = np.random.choice(range(self.n), self.n, replace=True)
            if self.classification:
                weak_learner = ClassificationTree(self.X[bootstrap_data], self.y[bootstrap_data], bootstrap_feature,
                                                  bootstrap_data, 0, self.max_depth, self.split_metric)
            else:
                weak_learner = RegressionTree(self.X[bootstrap_data], self.y[bootstrap_data], bootstrap_feature,
                                              bootstrap_data, 0, self.max_depth)
            self.weak_learners.append(weak_learner)

# if __name__ == '__main__':
#     data = pd.read_csv('../data/diabetes.csv', sep=',')
#     X = np.asarray(data.iloc[:, :-1])
#     y = np.asarray(data.iloc[:, -1])
#     model = RandomForest(split_metric='entropy', n_trees=10)
#     preds = model.fit(X, y)
#     mat = confusion_matrix(y, preds)
#     print(mat)

# if __name__ == '__main__':
#     data = pd.read_csv('../data/housing.csv', sep=',')
#     X = np.asarray(data.iloc[:, :-1])
#     y = np.asarray(data.iloc[:, -1])
#     model = RandomForest(classification=False, n_trees=5, max_depth=100)
#     preds = model.fit(X, y)
