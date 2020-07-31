import pandas as pd
import numpy as np
import sys
from criterions import *
from copy import deepcopy

sys.path.insert(0, '../02-Linear-Models/')
from regression import LinearRegression


class stepwise_selection:

    """
    For linear regression only.
    Iteration criterion: R2
    """

    def __init__(self, df, threshold):
        self.X = np.asarray(df.iloc[:, :-1])
        self.y = np.asarray(df.iloc[:, -1])
        self.p = df.shape[1] - 1
        self.best_r2 = 0
        self.threshold = threshold
        self.__feature_names = df.columns
        self.__best_features = []
        self.__initial_features = [i for i in range(self.p)]

    def forward_selection(self):
        for _ in range(self.p):
            iter_r2_increase = []
            for var_idx in self.__initial_features:
                current_r2 = self.__build_model(var_idx, 'forward')
                iter_r2_increase.append(current_r2 - self.best_r2)
            iter_r2_increase = np.asarray(iter_r2_increase)
            if max(iter_r2_increase) < self.threshold:
                return self.__feature_names[self.__best_features]
            else:
                selected_feature_idx = self.__initial_features.pop(iter_r2_increase.argmax())
                self.__best_features.append(selected_feature_idx)
                self.best_r2 += max(iter_r2_increase)
        return self.__feature_names[self.__best_features]

    def __build_model(self, var_idx, method='forward'):
        linear_reg = LinearRegression(gradient_descent=False)
        if method == 'forward':
            candidate_features = self.__best_features + [var_idx]
        elif method == 'backward':
            candidate_features = deepcopy(self.__best_features)
            candidate_features.remove(var_idx)
        X = self.X[:, candidate_features]
        linear_reg.fit(X, self.y)
        y_preds = [linear_reg.predict(x) for x in X]
        return calculate_r2(self.y, y_preds)

    def backward_selection(self):
        self.best_r2 = self.__get_full_model_r2()
        self.__best_features = self.__initial_features
        for _ in range(self.p):
            iter_r2_decrease = []
            for var_idx in self.__initial_features:
                current_r2 = self.__build_model(var_idx, 'backward')
                iter_r2_decrease.append(self.best_r2 - current_r2)
            iter_r2_decrease = np.asarray(iter_r2_decrease)
            if min(iter_r2_decrease) > self.threshold:
                return self.__feature_names[self.__best_features]
            else:
                self.__initial_features.pop(iter_r2_decrease.argmin())
                self.best_r2 -= min(iter_r2_decrease)
        return self.__feature_names[self.__best_features]

    def __get_full_model_r2(self):
        linear_reg = LinearRegression(gradient_descent=False)
        linear_reg.fit(self.X, self.y)
        y_preds = [linear_reg.predict(x) for x in self.X]
        return calculate_r2(self.y, y_preds)

    def bidirectional_selection(self):
        for _ in range(self.p):
            iter_r2_increase = []
            for var_idx in self.__initial_features:
                current_r2 = self.__build_model(var_idx, 'forward')
                iter_r2_increase.append(current_r2 - self.best_r2)
            iter_r2_increase = np.asarray(iter_r2_increase)
            if len(iter_r2_increase) > 0 and max(iter_r2_increase) > self.threshold:
                selected_feature_idx = self.__initial_features.pop(iter_r2_increase.argmax())
                self.__best_features.append(selected_feature_idx)
                self.best_r2 += max(iter_r2_increase)

            iter_r2_decrease = []
            for var_idx in self.__best_features:
                current_r2 = self.__build_model(var_idx, 'backward')
                iter_r2_decrease.append(self.best_r2 - current_r2)
            iter_r2_decrease = np.asarray(iter_r2_decrease)
            if len(iter_r2_decrease) > 0 and min(iter_r2_decrease) < self.threshold:
                self.__initial_features.pop(iter_r2_decrease.argmin())
                self.best_r2 -= min(iter_r2_decrease)

            if max(iter_r2_increase, default=0) < self.threshold < min(iter_r2_decrease,default=0):
                return self.__feature_names[self.__best_features]
        return self.__feature_names[self.__best_features]

if __name__ == '__main__':
    data = pd.read_csv('../data/housing.csv', sep=',')
    forward = stepwise_selection(data, 0.1)
    a = forward.forward_selection()
    backward = stepwise_selection(data, 0.5)
    b = backward.backward_selection()
    mix = stepwise_selection(data, 0.01)
    c = mix.bidirectional_selection()