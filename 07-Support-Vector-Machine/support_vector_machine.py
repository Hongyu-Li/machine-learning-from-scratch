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
        self.X, self.y = None, None
        self.n, self.p = 0, 0

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n, self.p = X.shape
        kernel_mat = self.__get_kernel_matrix()
        lagr_mult = self.__lagrange_solver(kernel_mat)
        self.__find_support_vectors(lagr_mult)
        self.__calculate_intercept()
        return [self.predict(x) for x in X]

    def __calculate_intercept(self):
        # Calculate intercept with the first support vector
        self.intercept = self.support_vectors_label[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vectors_label[i] * \
                              (self.kernel(self.support_vectors_feature[i], self.support_vectors_feature[0]))

    def __find_support_vectors(self, lagr_mult):
        support_vectors = lagr_mult > self.threshold
        self.lagr_multipliers = lagr_mult[support_vectors]
        self.support_vectors_feature = self.X[support_vectors]
        self.support_vectors_label = self.y[support_vectors]

    def __lagrange_solver(self, kernel_mat):
        cvxopt.solvers.options['show_progress'] = False
        P = cvxopt.matrix(np.outer(self.y, self.y) * kernel_mat, tc='d')
        q = cvxopt.matrix(np.ones(self.n) * -1, tc='d')
        A = cvxopt.matrix(self.y, (1, self.n), tc='d')
        b = cvxopt.matrix(0, tc='d')
        G = cvxopt.matrix(np.eye(self.n) * -1, tc='d')
        h = cvxopt.matrix(np.zeros(self.n), tc='d')
        if self.C:
            # stack constraints into one matrix
            G_slack = cvxopt.matrix(np.eye(self.n), tc='d')
            h_slack = cvxopt.matrix(np.ones(self.n) * self.C, tc='d')
            G = cvxopt.matrix(np.vstack((G, G_slack)), tc='d')
            h = cvxopt.matrix(np.vstack((h, h_slack)), tc='d')
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        lagr_mult = np.ravel(solution['x'])
        return lagr_mult

    def __get_kernel_matrix(self):
        kernel_mat = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                kernel_mat[i, j] = self.kernel(self.X[i], self.X[j])
        return kernel_mat

    def predict(self, X_new):
        y_pred = 0
        for i in range(len(self.lagr_multipliers)):
            y_pred += self.lagr_multipliers[i] * self.support_vectors_label[i] * \
                      (self.kernel(self.support_vectors_feature[i], X_new))
        y_pred += self.intercept
        return np.sign(y_pred)


class SVR:
    '''
    Ref: https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf
    https://github.com/fukuball/fuku-ml/blob/master/FukuML/SupportVectorRegression.py
    '''
    def __init__(self, threshold=1e-10, C=1, kernel=linear_kernel(), eps=1e-2):
        self.threshold = threshold
        self.C = C
        self.lagr_multipliers = None
        self.intercept = 0
        self.support_vectors_feature = None
        self.support_vectors_label = None
        self.kernel = kernel
        self.eps = eps
        self.X, self.y = None, None
        self.n, self.p = 0, 0

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n, self.p = X.shape
        kernel_mat = self.__get_kernel_matrix()
        lagr_mult = self.__lagrange_solver(kernel_mat)
        self.__find_support_vectors(lagr_mult)
        self.__calculate_intercept()
        return [self.predict(x) for x in X]

    def __find_support_vectors(self, lagr_mult):
        alpha_upper = lagr_mult[0]
        alpha_lower = lagr_mult[1]
        self.lagr_multipliers = alpha_upper - alpha_lower
        support_vectors = self.lagr_multipliers > self.threshold
        self.lagr_multipliers = self.lagr_multipliers[support_vectors]
        self.support_vectors_feature = self.X[support_vectors]
        self.support_vectors_label = self.y[support_vectors]

    def __calculate_intercept(self):
        # Calculate intercept with all support vectors
        inceptions = []
        for j in range(len(self.lagr_multipliers)):
            intercept = self.support_vectors_label[j] + self.eps
            for i in range(len(self.lagr_multipliers)):
                intercept -= self.lagr_multipliers[i] * \
                             (self.kernel(self.support_vectors_feature[i], self.support_vectors_feature[j]))
            inceptions.append(intercept)
        self.intercept = np.mean(inceptions)

    def __get_kernel_matrix(self):
        kernel_mat = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                kernel_mat[i, j] = self.kernel(self.X[i], self.X[j])
        return kernel_mat

    def __lagrange_solver(self, kernel_mat):
        cvxopt.solvers.options['show_progress'] = False
        P = cvxopt.matrix(
            np.vstack((np.hstack((kernel_mat, -kernel_mat)), np.hstack((-kernel_mat, kernel_mat)))),
            tc='d')
        q = cvxopt.matrix(np.hstack((self.eps - self.y, self.eps + self.y)), tc='d')
        A = cvxopt.matrix(np.hstack((np.ones(self.n), -np.ones(self.n))), (1, 2 * self.n), tc='d')
        b = cvxopt.matrix(0, tc='d')
        G = cvxopt.matrix(np.eye(2 * self.n) * -1, tc='d')
        h = cvxopt.matrix(np.zeros(2 * self.n), tc='d')
        G_slack = cvxopt.matrix(np.eye(2 * self.n), tc='d')
        h_slack = cvxopt.matrix(np.ones(2 * self.n) * self.C, tc='d')
        G = cvxopt.matrix(np.vstack((G, G_slack)), tc='d')
        h = cvxopt.matrix(np.vstack((h, h_slack)), tc='d')
        solution = cvxopt.solvers.qp(P, q, G, h)
        lagr_mult = np.ravel(solution['x']).reshape((2, -1))
        return lagr_mult

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