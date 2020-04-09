import numpy as np
import pandas as pd
from utils import confusion_matrix
import cvxopt
from kernels import linear_kernel, rbf_kernel, polynomial_kernel


class SVC:
    '''
    only implement svm classifier for binary classification
    '''
    def __init__(self, threshold=1e-10, C=None, kernel=linear_kernel()):
        self.threshold = threshold
        self.C = C
        self.lagr_multipliers = None
        self.intercept = 0
        self.support_vectors_feature = None
        self.support_vectors_label = None
        self.kernel = kernel

    def fit(self, X, y):
        n, p = X.shape
        cvxopt.solvers.options['show_progress'] = False
        P = cvxopt.matrix(np.outer(y, y) * (self.kernel(X, X)), tc='d')
        q = cvxopt.matrix(np.ones(n) * -1, tc='d')
        A = cvxopt.matrix(y, (1, n), tc='d')
        b = cvxopt.matrix(0, tc='d')
        G = cvxopt.matrix(np.eye(n) * -1, tc='d')
        h = cvxopt.matrix(np.zeros(n), tc='d')
        if self.C:
            # stack constraints into one matrix
            G_slack = cvxopt.matrix(np.eye(n), tc='d')
            h_slack = cvxopt.matrix(np.ones(n) * self.C, tc='d')
            G = cvxopt.matrix(np.vstack((G, G_slack)), tc='d')
            h = cvxopt.matrix(np.vstack((h, h_slack)), tc='d')
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        lagr_mult = np.ravel(solution['x'])
        support_vectors = lagr_mult > self.threshold
        self.lagr_multipliers = lagr_mult[support_vectors]
        self.support_vectors_feature = X[support_vectors]
        self.support_vectors_label = y[support_vectors]

        # Calculate intercept with the first support vector
        self.intercept = self.support_vectors_label[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vectors_label[i] * \
                              (self.kernel(self.support_vectors_feature[i], self.support_vectors_feature[0]))
        return [self.predict(x) for x in X]

    def predict(self, X_new):
        y_pred = 0
        for i in range(len(self.lagr_multipliers)):
            y_pred += self.lagr_multipliers[i] * self.support_vectors_label[i] * \
                      (self.kernel(self.support_vectors_feature[i], X_new))
        y_pred += self.intercept
        return np.sign(y_pred)


# if __name__ == '__main__':
#     data = pd.read_csv('../data/diabetes.csv', sep=',')
#     X = np.asarray(data.iloc[:, :-1])
#     y = np.asarray(data.iloc[:, -1])
#     y[y == 0] = -1
#     # model = SVC(C=2, threshold=1e-8, kernel=rbf_kernel(0.1))
#     model = SVC(C=2, threshold=1e-8, kernel=polynomial_kernel(2))
#     preds = model.fit(X, y)
#     mat = confusion_matrix(y, preds)
#     print(mat)
