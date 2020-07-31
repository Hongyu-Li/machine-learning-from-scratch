import numpy as np
import pandas as pd
import itertools
from criterions import *
import matplotlib.pyplot as plt
from regression import LinearRegression


class best_subset_selection:
    """
    For linear regression only.
    """
    def __init__(self):
        self.y = None
        self.n, self.p = 0, 0
        self.models_summary = None
        self.selected_features = None

    def build_models(self, df):
        self.n, self.p = df.shape
        performances = []
        for k in range(1, self.p):
            for var_combo in itertools.combinations(df.columns[:-1], k):
                linear_reg = LinearRegression()
                X = np.asarray(df[list(var_combo)])
                self.y = np.asarray(df.iloc[:, -1])
                linear_reg.fit(X, self.y)
                y_preds = [linear_reg.predict(x) for x in X]
                adj_r2, aic, bic, r2, rss = self.__calculate_criterions(y_preds, k)
                performance = [var_combo, k, aic, bic, rss, r2, adj_r2]
                performances.append(performance)
                col_names = ['subset', 'num_of_variables', 'aic', 'bic', 'rss', 'r2',
                             'adj_r2']
                self.models_summary = pd.DataFrame(performances, columns=col_names)
        self.__visualize_best_subset_performance()

    def get_selection(self, criterions=['rss', 'aic']):
        train_err_criterion, test_err_criterion = criterions
        if train_err_criterion not in ['rss', 'r2'] or test_err_criterion not in ['aic', 'bic', 'adj_r2']:
            raise Exception('Please input correct parameters in the arguments.')
        else:
            flag_name = 'min_rss' if train_err_criterion == 'rss' else 'max_r2'
            round1_selection = self.models_summary[
                self.models_summary[train_err_criterion] == self.models_summary[flag_name]]
            if test_err_criterion == 'adj_r2':
                selected_ind = round1_selection[test_err_criterion].idxmax()
            else:
                selected_ind = round1_selection[test_err_criterion].idxmin()
            self.selected_features = round1_selection.iloc[selected_ind, 0]
            print('=========The Best Subset Is===========\n' + str(self.selected_features))

    def __visualize_best_subset_performance(self):

        self.models_summary['min_rss'] = self.models_summary.groupby('num_of_variables')['rss'].transform(min)
        self.models_summary['max_r2'] = self.models_summary.groupby('num_of_variables')['r2'].transform(max)
        plt.subplot(1, 2, 1)
        plt.scatter(self.models_summary['num_of_variables'], self.models_summary['rss'])
        plt.xlabel('number of variables')
        plt.ylabel('RSS')
        plt.plot(self.models_summary['num_of_variables'], self.models_summary['min_rss'],
                 color="r", label="Best subset")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.scatter(self.models_summary['num_of_variables'], self.models_summary['r2'])
        plt.xlabel('number of variables')
        plt.ylabel('R^2')
        plt.plot(self.models_summary['num_of_variables'], self.models_summary['max_r2'],
                 color="r", label="Best subset")
        plt.legend()
        plt.show()

    def __calculate_criterions(self, y_preds, k):
        rss = calculate_rss(self.y, y_preds)
        r2 = calculate_r2(self.y, y_preds)
        adj_r2 = calculate_adjust_r2(self.y, y_preds, k)
        aic = calculate_aic(self.y, y_preds, k)
        bic = calculate_bic(self.y, y_preds, k)
        return adj_r2, aic, bic, r2, rss


if __name__ == '__main__':
    data = pd.read_csv('../data/housing.csv', sep=',')
    selection = best_subset_selection()
    selection.build_models(data)
    selection.get_selection()
