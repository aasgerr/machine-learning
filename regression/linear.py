import numpy as np
from scipy.optimize import minimize


class GDLinRegress:
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
                    sum((w.dot(x[i, :]) + b - y[i]) * (x[i, j]) for i in range(m)) / m
                )

            return np.array(dw), sum((w.dot(x[i, :]) + b - y[i]) / m for i in range(m))

        def gd(x, y, w, b):
            for i in range(self.iter):
                dw, db = grad(x, y, w, b)
                w -= self.alpha * dw
                b -= self.alpha * db

            return w, b

        self.w, self.b = gd(Xtrain, ytrain, np.zeros(n), 0)

    def predict(self, Xtest):
        return Xtest.dot(self.w) + self.b


class NormLinRegress:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, Xtrain, ytrain):
        # X_train: (m, n)
        # y_train: (m, )
        # m = samples, n = features

        m, n = np.shape(Xtrain)
        Xb = np.c_[np.ones((m, 1)), Xtrain]
        self.w = np.linalg.inv(Xb.T.dot(Xb)).dot(Xb.T).dot(ytrain)
        self.b = self.w[0]
        self.w = self.w[1:]

    def predict(self, Xtest):
        return Xtest.dot(self.w) + self.b


class QRLinRegress:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, Xtrain, ytrain):
        # X_train: (m, n)
        # y_train: (m, )
        # m = samples, n = features

        m, n = np.shape(Xtrain)
        Xb = np.c_[np.ones((m, 1)), Xtrain]
        Q, R = np.linalg.qr(Xb)
        self.w = np.linalg.inv(R).dot(Q.T).dot(ytrain)
        self.b = self.w[0]
        self.w = self.w[1:]

    def predict(self, Xtest):
        return Xtest.dot(self.w) + self.b


class SVDLinRegress:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, Xtrain, ytrain):
        # X_train: (m, n)
        # y_train: (m, )
        # m = samples, n = features

        m, n = np.shape(Xtrain)
        Xb = np.c_[np.ones((m, 1)), Xtrain]
        self.w = np.linalg.pinv(Xb).dot(ytrain)
        self.b = self.w[0]
        self.w = self.w[1:]

    def predict(self, Xtest):
        return Xtest.dot(self.w) + self.b


class scipyRegress:
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
            ypred = Xb.dot(w)
            return np.mean((ypred - ytrain) ** 2)

        result = minimize(objective, np.zeros(n + 1), tol=1e-10)
        self.w = result.x
        self.b = self.w[0]
        self.w = self.w[1:]

    def predict(self, Xtest):
        return Xtest.dot(self.w) + self.b
