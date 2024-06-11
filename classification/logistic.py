import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

def g(z):
    return 1 / (1 + np.exp(-z))


class GDLogRegress:
    def __init__(self, alpha=0.1, iter=10000):
        self.w = None
        self.b = None
        self.alpha = alpha
        self.iter = iter

    def fit(self, Xtrain, ytrain):
        # X_train: (m, n)
        # y_train: (m, )
        # m = samples, n = features

        m, n = np.shape(Xtrain)

        def grad(x, y, w, b):
            dw = []
            for j in range(n):
                dw.append(
                    sum((g(w.dot(x[i, :]) + b) - y[i]) * (x[i, j]) for i in range(m)) / m
                )

            return np.array(dw), sum((g(w.dot(x[i, :]) + b) - y[i]) / m for i in range(m))

        def gd(x, y, w, b):
            for i in range(self.iter):
                dw, db = grad(x, y, w, b)
                w -= self.alpha * dw
                b -= self.alpha * db

            return w, b

        self.w, self.b = gd(Xtrain, ytrain, np.ones(n)*0.5, 0.5)

    def predict(self, Xtest):
        return g(Xtest.dot(self.w) + self.b)


class scipyLogRegress:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, Xtrain, ytrain):
        # X_train: (m, n)
        # y_train: (m, )
        # m = samples, n = features

        m, n = np.shape(Xtrain)
        Xb = np.c_[np.ones((m, 1)), Xtrain]

        def objective(w):
            ypred = g(Xb.dot(w))
            return np.mean(ytrain*np.log(ypred) + (1 - ytrain)*np.log(1 - ypred))

        result = minimize(objective, np.ones(n + 1)*0.5, tol=1e-10)
        self.w = result.x
        self.b = self.w[0]
        self.w = self.w[1:]

    def predict(self, Xtest):
        return g(Xtest.dot(self.w) + self.b)
