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

        kernel_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                kernel_mat[i,j]=self.kernel(X[i], X[j])

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


class SVR:
    def __init__(self, threshold=1e-10, C=None, kernel=linear_kernel(), eps=1e-2):
        self.threshold = threshold
        self.C = C
        self.lagr_multipliers = None
        self.intercept = 0
        self.support_vectors_feature = None
        self.support_vectors_label = None
        self.kernel = kernel
        self.eps = eps

    def fit(self, X, y):
        n, p = X.shape
        cvxopt.solvers.options['show_progress'] = False

        kernel_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                kernel_mat[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(
            np.vstack((np.hstack((kernel_mat, -kernel_mat)), np.hstack((-kernel_mat, kernel_mat)))),
            tc='d')
        q = cvxopt.matrix(np.hstack((self.eps - y, self.eps + y)), tc='d')
        A = cvxopt.matrix(np.hstack((np.ones(n), -np.ones(n))), (1, 2*n), tc='d')
        b = cvxopt.matrix(0, tc='d')
        G = cvxopt.matrix(np.eye(2*n) * -1, tc='d')
        h = cvxopt.matrix(np.zeros(2*n), tc='d')
        G_slack = cvxopt.matrix(np.eye(2*n), tc='d')
        h_slack = cvxopt.matrix(np.ones(2*n) * self.C, tc='d')
        G = cvxopt.matrix(np.vstack((G, G_slack)), tc='d')
        h = cvxopt.matrix(np.vstack((h, h_slack)), tc='d')
        solution = cvxopt.solvers.qp(P, q, G, h)
        lagr_mult = np.ravel(solution['x']).reshape((2, -1))
        alpha_upper= lagr_mult[0]
        alpha_lower = lagr_mult[1]
        self.lagr_multipliers = alpha_upper - alpha_lower
        support_vectors = self.lagr_multipliers > self.threshold
        self.lagr_multipliers = self.lagr_multipliers[support_vectors]
        self.support_vectors_feature = X[support_vectors]
        self.support_vectors_label = y[support_vectors]

        # Calculate intercept with all support vectors
        inceptions = []
        for j in range(len(self.lagr_multipliers)):
            intercept = self.support_vectors_label[j] + self.eps
            for i in range(len(self.lagr_multipliers)):
                intercept -= self.lagr_multipliers[i] * \
                                (self.kernel(self.support_vectors_feature[i], self.support_vectors_feature[j]))
            inceptions.append(intercept)
        self.intercept = np.mean(inceptions)
        return [self.predict(x) for x in X]

    def predict(self, X_new):
        y_pred = 0
        for i in range(len(self.lagr_multipliers)):
            y_pred += self.lagr_multipliers[i] * \
                      (self.kernel(self.support_vectors_feature[i], X_new))
        y_pred += self.intercept
        return y_pred


# if __name__ == '__main__':
#     data = pd.read_csv('../data/diabetes.csv', sep=',')
#     X = np.asarray(data.iloc[:, :-1])
#     y = np.asarray(data.iloc[:, -1])
#     y[y == 0] = -1
#     model = SVC(C=2, threshold=1e-8, kernel=rbf_kernel(0.1))
#     # model = SVC(C=2, threshold=1e-8, kernel=polynomial_kernel(2))
#     preds = model.fit(X, y)
#     mat = confusion_matrix(y, preds)
#     print(mat)


# if __name__ == '__main__':
#     rng = np.random.RandomState(0)
#     X = np.sort(5 * np.random.rand(40, 1), axis=0)
#     y = np.sin(X).ravel()
#     y[::5] += 3 * (0.5 - np.random.rand(8))
#     model = SVR(C=10, threshold=1e-6, eps=0.001, kernel=rbf_kernel(0.001))
#     preds = model.fit(X, y)