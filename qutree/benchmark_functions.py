import numpy as np


def V(x):
    N = x.shape[0]
    v = 0.
    for i in range(N):
        v += 0.5*(x[i])**2 + 0.1*x[i]*x[(i+1) % N]
    return v


def filterV(x):
    return np.exp(-V(x))


def pot(x):
    v = 0.
    for i in range(0, x.shape[0]):
        v += np.sin(np.pi*x[i]**2)/(x[i]**2+1e-7)
    for i in range(0, x.shape[0]):
        v+=np.sin(x[i]*x[i-1])
    return v - 15.


def rastigrin(x):
    y = 5.12 * (x - 0.5) * 2
    A = 10
    n = len(x)
    v = n * A
    for i in range(n):
        v += (y[i]**2 - A * np.cos(np.pi * 2 * y[i]))
    return v


def double_minimum(x):
    x = np.asarray(x)
    return np.sum((x - 0.1)**2 * (x - 0.8)**2) + 1


def double_minimum_nd(x):
    x = np.asarray(x)
    well1 = np.sum((x - 0.1)**2) + 0.1
    well2 = np.sum((x - 0.8)**2)
    return np.minimum(well1, well2) + 0.1 * well1


def double_minimum_exp(x):
    x = np.asarray(x)
    well1 = (np.exp(-np.sum(3*(x - 0.1)**2))-1)
    well2 = (np.exp(-np.sum(5*(x - 0.8)**2)))*0.8
    return -(well1 + well2)


def rosenbrock_function(x):
    return np.sum(100. * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

