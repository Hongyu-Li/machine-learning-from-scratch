import numpy as np
from activations import Sigmoid, Tanh, Linear, Softmax
from utils import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from losses import MSE, LogLoss

class NN:
    '''
    Perceptron with one hidden layer, BGD to train
    '''
    def __init__(self, n_iters=10000, hidden_activation=Sigmoid(), output_activation=Linear(), learning_rate=1e-2,
                 n_hidden=10, loss=MSE(), mini_batch=10):
        self.n_iters = n_iters
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.W = None
        self.W0 = None
        self.V = None
        self.V0 = None
        self.mini_batch = mini_batch
        self.loss = loss
        self.X, self.y = None, None
        self.n_hidden = n_hidden

    def fit(self, X, y):
        np.random.seed(0)
        n, p = X.shape
        # one hot encoding
        _, n_outputs = y.shape
        self.W = np.random.uniform(size=p*self.n_hidden).reshape((p, self.n_hidden))
        self.W0 = np.random.uniform(size=self.n_hidden)
        self.V = np.random.uniform(size=(self.n_hidden, n_outputs))
        self.V0 = np.random.uniform(size=n_outputs)
        self.X = X
        self.y = y
        for i in range(self.n_iters):
            samples_idx = np.random.choice(range(n), size=self.mini_batch, replace=False)
            linear_output = self.X[samples_idx].dot(self.W) + self.W0
            hidden_output = self.hidden_activation(linear_output)
            y_pred_input = hidden_output.dot(self.V) + self.V0
            y_pred = self.output_activation(y_pred_input)
            grad = self.loss.gradient(self.y[samples_idx], y_pred) * self.output_activation.gradient(y_pred_input)
            grad_wrt_V = hidden_output.T.dot(grad)
            grad_wrt_V0 = np.sum(grad, axis=0)
            grad_wrt_hidden_layer = grad.reshape(-1, n_outputs).dot(self.V.reshape(n_outputs, -1)) * \
                                    self.hidden_activation.gradient(linear_output)
            grad_wrt_W = self.X[samples_idx].T.dot(grad_wrt_hidden_layer)
            grad_wrt_W0 = np.sum(grad_wrt_hidden_layer, axis=0)
            self.V -= self.learning_rate * grad_wrt_V
            self.V0 -= self.learning_rate * grad_wrt_V0
            self.W -= self.learning_rate * grad_wrt_W
            self.W0 -= self.learning_rate * grad_wrt_W0
        return np.ravel([self.predict(x) for x in self.X])

    def predict(self, X_new):
        hidden_output = self.hidden_activation(X_new.dot(self.W) + self.W0)
        if self.loss.__class__ == MSE:
            return self.output_activation(hidden_output.dot(self.V) + self.V0)
        else:
            return np.argmax(self.output_activation(hidden_output.dot(self.V) + self.V0), axis=0)

# if __name__ == '__main__':
#     rng = np.random.RandomState(0)
#     X = np.sort(5 * np.random.rand(40, 1), axis=0)
#     y = np.sin(X).ravel()
#     y[::5] += 3 * (0.5 - np.random.rand(8))
#     y = y.reshape(-1,1)
#     model = NN(learning_rate=1e-3, hidden_activation=Tanh(), n_iters=100000, mini_batch=X.shape[0])
#     preds = model.fit(X, y)

# if __name__ == '__main__':
#     data = pd.read_csv('../data/diabetes.csv', sep=',')
#     X = data.iloc[:, :-1]
#     normalized_df = (X - X.mean()) / X.std()
#     X = np.asarray(normalized_df)
#     y = np.asarray(data.iloc[:, -1])
#     nb_classes = len(np.unique(y))
#     targets = y.reshape(-1)
#     one_hot_targets = np.eye(nb_classes)[targets]
#     model = NN(learning_rate=1e-4, loss=LogLoss(), hidden_activation=Sigmoid(), output_activation=Softmax(),
#                n_iters=10000, mini_batch=30)
#     preds = model.fit(X, one_hot_targets)
#     mat = confusion_matrix(y, preds)
#     print(mat)