import numpy as np



def x1D(xa, xe, N):
    x = np.ndarray([N])
    dx = (xe - xa) / (N - 1)
    for I in range(N):
        x[I] = xa + I * dx
    return x

def x1Dto3D(x):
    N = len(x)
    z = np.zeros([N])
    return np.vstack((x, z, z)).T

def eval1D(f, n: int, interval: list):
    """
    * f: function to evaluate
    * n: number of points
    * interval: [xa, xe]
    """
    N = 2**n
    A = np.ndarray([N])
    x = np.ndarray([N])
    xa, xe = interval
    X = x1D(xa, xe, N)
    for I, x in enumerate(X):
        A[I] = f(x)
    return X, A