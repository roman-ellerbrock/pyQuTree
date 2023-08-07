import numpy as np


def n():
    return np.array([[0, 0], [0, 1]])


def x():
    return np.array([[0, 1], [1, 0]])


def z():
    return np.array([[1, 0], [0, -1]])


def a():
    return np.array([[0, 1], [0, 0]])


def adagger():
    return np.array([[0, 0], [1, 0]])


def jump(site0, site1):
    M = []
    M.append([a, site0])
    M.append([adagger, site1])
    return M


def N(size):
    H = []
    for i in range(size):
        H.append([n(), i])
    return H


def IsingTransversal(Nl, gamma):
    H = []
    for i in range(0, Nl-1):
        H.append(jump(i, i+1))
        H.append(jump(i+1, i))

    for i in range(0, Nl):
        H.append([gamma * z(), i])

    return H



def apply(A, H):
    for pair in H:
        