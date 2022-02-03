import numpy as np
import pandas as pd

class LogisticRegression():
    def __init__(self,
                 solver = 'irls',
                 max_iter = 100):
        if not isinstance(solver, str):
            raise ValueError("solver must be str format")
        if not isinstance(max_iter, int):
            raise ValueError("max_iter must be int format")
        if solver not in ['irls']:
            raise ValueError('solver {} is not specified'.format(solver))
        self.solver = solver
        self.max_iter = max_iter
        self.weights = None
        
    def _log_loss(self, labels, y_pred):
        return np.sum(-labels * np.log(y_pred) - (1 - labels) * np.log(1 - y_pred), axis = 0) / len(labels)
    
    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    
    def _irls_optimization(self, train_data, train_labels, step):
        w = np.linalg.inv(train_data.T@train_data)@train_data.T@train_labels.values
        
        for t in range(self.max_iter):
            sigm_list = []
            new_sigm = []
            gams = []
            y_mod = []
            # Вычисляю сигмы
            for i in range(len(train_labels)):
                sigm_list.append(self._sigmoid(w.T@train_data[i]*train_labels.values[i][0])[0])
        
            # Вычисляю гаммы
            for j in range(len(train_labels)):
                gams.append(np.sqrt((1-sigm_list[j])/sigm_list[j]))
        
            # Матрица взвешенных объектов
            F = np.diag(gams)@train_data
    
            # Вектор модернизированных ответов
            for k in range(len(train_labels)):
                y_mod.append(train_labels.values[k][0] * np.sqrt((1 - sigm_list[k]) / sigm_list[k]))
        
            # Теперь решается задача лин.регрессии со взвешенными объектами и модернизированными ответами
            w = w + step * np.linalg.inv(F.T@F)@F.T@np.array([y_mod]).T  # Новый вектор весов
        
        return w
    
    def fit(self, train_data, train_labels, step = 1):
        if self.solver == 'irls':
            self.weights = self._irls_optimization(train_data, train_labels, step)
    
    def predict_proba(self, X):
        return self._sigmoid(X@self.weights)
    
    def predict(self, X, treshold = 0.5):
        ret = np.array([1 if elem>treshold else 0  for index, elem in enumerate(self.predict_proba(X))])
        return ret