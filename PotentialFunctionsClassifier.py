import numpy as np
import pandas as pd

from kernels import K, K_gauss, K_triangle
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt



class PFM(object):
    def __init__(self, kernel = None):
        self.X, self.Y, self.gams = None, None, None
        self.kernel = lambda x: np.ones_like(x)
        if kernel is not None:
            self.kernel = kernel
    def score(self, X):
        # Размерности: X=(1, кол-во признаков), self.X=(кол-во объектов обуч.выборки, кол-во признаков),
        # self.gams=(кол-во объектов обуч.выборки, ) => w=(1, кол-во объектов) - вектор весов объектов
        w = self.gams.reshape((len(self.Y),)) * self.kernel(cdist(X, self.X))
        sum_of_weights_for_zero_class_objects = np.sum(w.T[np.where(self.Y == 0)[0]].T, axis = -1)
        sum_of_weights_for_one_class_objects = np.sum(w.T[np.where(self.Y == 1)[0]].T, axis = -1)
        scores = np.vstack(
            [sum_of_weights_for_zero_class_objects,
            sum_of_weights_for_one_class_objects]).T
        return scores
    
    def predict(self, X):
        # возвращает номер класса, у которого наибольшая сумма весов
        return np.argmax(self.score(X), axis = -1)
    def fit(self, X, Y, epochs = 10):
        # Инициализирую X, Y. Гамму инициализирую массивом нулей размерностью Y
        self.X, self.Y, self.gams = np.array(X), np.array(Y).reshape((np.array(Y).shape[0],)), np.zeros_like(Y.values.reshape((Y.shape[0],)))
        for _ in range(epochs):
            # для каждой пары (объект, ответ): если предсказание неверно, значение соответствующей гаммы увел. на единицу
            for i, (x, y) in enumerate(zip(self.X, self.Y)):
                if self.predict(np.array([x]))[0] != y:
                    self.gams[i] += 1
                    
    def margin_distribution(self, X, y):
        M = [s[y] -s[y-1] for s, y in zip(self.score(X), y.values)]
        # Сортировка массива отступов по возрастанию
        M = np.array(sorted(M)).reshape((len(M),))
        
        x = list(range(len(M)))
        plt.figure(figsize = (8, 6))
        plt.title('График отступов')
        plt.xlabel('Объект')
        plt.ylabel('Отступ')
        plt.plot(x, M)

        plt.hlines(0, 0, len(x), color = 'black')
        plt.fill_between(x, M, where = M<0, color = 'red', alpha = 0.5, label = 'Неверно классифицированные объекты')
        plt.fill_between(x, M, where = M>0, color = 'green', alpha = 0.5, label = 'Верно классифицированные объекты')
        #plt.xlim(0, 300)
        plt.legend()
        plt.show()