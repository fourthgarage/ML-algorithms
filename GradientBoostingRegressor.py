import numpy as np
import pandas as pd
from DecisionTreeRegressor import DecisionTreeRegressor

class GradientBoostingRegressor():
    def __init__(self, base_estimator = None, n_estimators = 10):
        self.n_estimators = n_estimators
        self.base_estimator = DecisionTreeRegressor(max_depth = 1)
        if base_estimator:
            self.base_estimator = base_estimator
        
        #self.b = [base.clone(self.base_estimator) for _ in range(self.n_estimators)]
        self.b = [base_estimator for _ in range(self.n_estimators)]
        
    def get_params(self, deep = True):
        return {
            'base_estimator': self.base_estimator,
            'n_estimators': self.n_estimators}
    def score(self, X, y):
        return ((self.predict(X) - y)**2).mean()
        
    def fit(self, X, y, cat_features = []):
        residual = y.copy()
        for algo in range(len(self.b)):
            self.b[algo].fit(X, residual, cat_features)
            residual = residual - self.b[algo].predict(X)
                
    def predict(self, X):
        return np.sum([elem.predict(X) for elem in self.b], axis=0)