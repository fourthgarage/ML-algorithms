import numpy as np
# Ядро используется квадратичное (1 - r**2). В случае, если r>1, то K = 0
def K(distance, h = 0.05):
    ret = np.array(distance)/h
    return (1 - ret**2) * (np.abs(ret) <= 1) # (1 - r**2) умножить на бинарную ф-ию. Если r>1, то пропускаем, иначе умножаем

# Инфинитное гауссовское ядро
def K_gauss(distance, h = 0.05):
    ret = np.array(distance)/h
    return (np.exp(-2 * ret**2))

# Финитное треугольное ядро
def K_triangle(distance, h = 0.05):
    ret = np.array(distance)/h
    return (1 - np.abs(ret)) * (np.abs(ret) <= 1)