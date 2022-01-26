import pandas as pd
import numpy as np
from tree import DecisionTreeClassifier


# ВАЖНО: в метод fit,predict передавать X в виде pd.DataFrame, y в виде pd.Series
class AdaBoostClassifier(object): 
    def __init__(self,
                 base_estimator = DecisionTreeClassifier(),
                 n_estimators = 100,
                 n_steps = 10):
        if not isinstance(n_estimators, int):
            raise ValueError("n_estimators must be int format")
        if not isinstance(n_steps, int):
            raise ValueError("n_steps must be int format")
        
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_steps = n_steps
        self.X, self.y, self.W, self.bt, self.best_tree, self.adaboost_tree = None,None,None,None,None,None
    
    
    
    
    def _get_classification(self, X, y, W, cat_features=[]):
        m,n = X.shape
        best_tree = {}
        # Обучение дерева
        tree = self.base_estimator
        tree.fit(X, y, cat_features, sample_weights=W)
        # Предсказание
        predict_values = tree.predict(X)
        # Определяем вектор ошибок
        vector_of_errs = pd.Series(np.ones(m,))
        # Если пресказание верное, то ошибка = 0
        vector_of_errs.at[predict_values == y] = 0
        # Кол-во ложных ответов, взятых с весом W
        N = (W * vector_of_errs).sum()
        best_tree['model'] = tree.tree
        return best_tree, N, predict_values
      
    
    def fit(self, X, y, n_estimators = 100, cat_features=[]):
        self.adaboost_tree = []
        self.X = X
        self.y = y
        m,n = X.shape
        # Инициализация весов
        self.W = pd.Series(data=np.ones((m,))/m)
        ensemble = pd.Series(data=np.zeros((m,)))
        for algo in range(self.n_estimators):
            # Словарь. Содержит базовый алгоритм и его вес
            params = {}
            # Обучаем базовый алгоритм
            best_tree, N, bt = self._get_classification(self.X, self.y, self.W, cat_features)
            self.y = self.y.astype('float32')
            bt = bt.astype('float32')
            #best_tree, N, bt = self._build_decision_stump(self.X, self.y, self.W)
            # Вычисляем альфу(вес алгоритма)
            alpha = 0.5 * np.log((1 - N + 1/m)/(N + 1/m))
            best_tree['alpha'] = alpha
            self.adaboost_tree.append(best_tree)
            
            # Замена 0 на -1 в векторах прогнозов и истинных ответов(для корректного подсчета отступов и весов объектов)
            bt.at[bt == 0] = -1
            self.y.at[self.y == 0] = -1
            # Обновляем веса объектов
            self.W = self.W * np.exp(-alpha*self.y*bt)
            # нормируем веса
            self.W = self.W/self.W.sum()
            
            # Расчет ошибки всех классификаторов. Если она нулевая - прекратить обучение
            ensemble += alpha * bt
            
            # Индикаторный вектор ошибок. Если прогноз не равен истинному y, то на нем ошибка(True)
            #errs = np.multiply(np.sign(ensemble) != y, np.ones((m,1)))
            
            errs = np.multiply(np.sign(ensemble) != self.y, pd.Series(data=np.ones((m,))))
            # Средняя ошибка классификации
            average_err = errs.sum()/m
            
            # Обратное преобразование вектора истинных ответов для корректной работы решающего дерева(замена -1 на 0)
            self.y.at[self.y == -1] = 0
            
            if average_err == 0.0:
                break
             
            # Если последние 3 итерации модели не менялись, то останавливаем работу алгоритма
            if len(self.adaboost_tree) > 3 and self.adaboost_tree[-1]['model'] == self.adaboost_tree[algo-2]['model']:
                break
      
    def predict(self, X):
        m = X.shape[0]
        weighted_voting = pd.Series(np.zeros((m,)), index=X.index.tolist())
        for algo in range(len(self.adaboost_tree)):
            bt = X.apply(self.base_estimator._predict_one_example, axis=1, args = (adaboost.adaboost_tree[algo]['model'],))
            bt = bt.astype('float32')
            bt.at[bt==0] = -1
            #print(np.unique(bt, return_counts=True))
            #print(bt)
            weighted_voting += self.adaboost_tree[algo]['alpha'] * bt
            
        ax = np.sign(weighted_voting)
        ax.at[ax == -1] = 0 
        return ax
    def accuracy(self, y_pred, y_true):
        accuracy = pd.Series(data=np.zeros((y_true.shape[0],)), index=y_true.index.tolist())
        y_pred.index = y_true.index.tolist()
        accuracy.at[y_pred == y_true] = 1
        return accuracy.sum()/len(accuracy)
    
    def _get_confusion_matrix(self, y_pred, y_true):
        TP = sum([1 if (y_pred == 1 and y_pred == y_true) else 0 for y_pred,y_true in zip(y_pred, y_true)])
        FP = sum([1 if (y_pred == 1 and y_pred != y_true) else 0 for y_pred,y_true in zip(y_pred, y_true)])
        TN = sum([1 if (y_pred == 0 and y_pred == y_true) else 0 for y_pred,y_true in zip(y_pred, y_true)])
        FN = sum([1 if (y_pred == 0 and y_pred != y_true) else 0 for y_pred,y_true in zip(y_pred, y_true)])
        return TP, FP, TN, FN
    
    def precision_score(self, y_pred, y_true):
        TP, FP, TN, FN = self._get_confusion_matrix(y_pred, y_true)
        return TP/(TP + FP)
        
    def recall_score(self, y_pred, y_true):
        TP, FP, TN, FN = self._get_confusion_matrix(y_pred, y_true)
        return TP/(TP + FN)
    
    def sensitivity_score(self, y_pred, y_true):
        TP, FP, TN, FN = self._get_confusion_matrix(y_pred, y_true)
        return TP/(TP + FN)
    def specificity_score(self, y_pred, y_true):
        TP, FP, TN, FN = self._get_confusion_matrix(y_pred, y_true)
        return TN/(TN + FP)