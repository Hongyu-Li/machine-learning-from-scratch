import pandas as pd
import numpy as np
import sys
from criterions import *
from copy import deepcopy
import itertools

sys.path.insert(0, '../02-Linear-Models/')
from regression import LinearRegression


class stepwise_selection:

    """
    For linear regression only.
    """

    def __init__(self):
        self.X, self.y = None, None

    def forward_selection(self, df, threshold, criterion):
        _, p = df.shape
        y = np.asarray(df.iloc[:, -1])
        flag = 0 if criterion == 'r2' else calculate_sst(y)
        selected_idx = []
        unselected_idx = [i for i in range(p-1)]
        for iter in range(p-1):
            iter_summary = []
            for var_idx in unselected_idx:
                linear_reg = LinearRegression(gradient_descent=False)
                candidate_idx = selected_idx + [var_idx]
                X = np.asarray(df.iloc[:, candidate_idx])
                linear_reg.fit(X, y)
                y_preds = [linear_reg.predict(x) for x in X]
                if criterion == 'rss':
                    candidate_flag = calculate_rss(y, y_preds)
                    iter_summary.append(candidate_flag)
                else:
                    candidate_flag = calculate_r2(y, y_preds)
                    iter_summary.append(candidate_flag)
            iter_summary = np.asarray(iter_summary)
            if max(abs(iter_summary - flag)) < threshold:
                return df.columns[selected_idx]
            else:
                selected_var_idx = abs(iter_summary - flag).argmax()
                selected_idx.append(selected_var_idx)
                unselected_idx.pop(selected_var_idx)
                flag = iter_summary[selected_var_idx]
        return df.columns[selected_idx]

    def backward_selection(self, df, threshold, criterion):
        _, p = df.shape
        y = np.asarray(df.iloc[:, -1])
        linear_reg = LinearRegression(gradient_descent=False)
        X = np.asarray(data)
        linear_reg.fit(X, y)
        y_preds = [linear_reg.predict(x) for x in X]
        flag = calculate_r2(y, y_preds) if criterion == 'r2' else calculate_rss(y, y_preds)
        selected_idx = [i for i in range(p - 1)]
        unselected_idx = []
        for iter in range(1, p-1):
            iter_summary = []
            for var_idx in selected_idx:
                linear_reg = LinearRegression(gradient_descent=False)
                candidate_idx = deepcopy(selected_idx)
                candidate_idx.remove(var_idx)
                X = np.asarray(data.iloc[:, candidate_idx])
                linear_reg.fit(X, y)
                y_preds = [linear_reg.predict(x) for x in X]
                if criterion == 'rss':
                    candidate_flag = calculate_rss(y, y_preds)
                    iter_summary.append(candidate_flag)
                else:
                    candidate_flag = calculate_r2(y, y_preds)
                    iter_summary.append(candidate_flag)
            iter_summary = np.asarray(iter_summary)
            if min(abs(iter_summary - flag)) < threshold:
                return df.columns[selected_idx]
            else:
                unselected_var_idx = abs(iter_summary - flag).argmin()
                unselected_idx.append(unselected_var_idx)
                selected_idx.pop(unselected_var_idx)
                flag = iter_summary[unselected_var_idx]
        return df.columns[selected_idx]

if __name__ == '__main__':
    data = pd.read_csv('../data/housing.csv', sep=',')
    selection = stepwise_selection()
    # a = selection.forward_selection(data, 1e-2, 'rss')
    b = selection.backward_selection(data, 1e-2, 'rss')