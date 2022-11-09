from sklearn.linear_model import Ridge, Lasso
import numpy as np


class RidgeClassifier(Ridge):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, X):
        pred = super().predict(X)
        a = [1 if p >= 0.5 else 0 for p in pred]
        return a

    def predict_cont(self, X):
        return super().predict(X)

    def score(self, X, y, sample_weight=None):
        a = np.mean(y == self.predict(X))
        return a


class LassoClassifier(Lasso):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, X):
        pred = super().predict(X)
        a = [1 if p >= 0.5 else 0 for p in pred]
        return a

    def predict_cont(self, X):
        return super().predict(X)

    def score(self, X, y, sample_weight=None):
        a = np.mean(y == self.predict(X))
        return a
