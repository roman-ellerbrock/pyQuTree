import numpy as np
from .ttn.grid import linspace

def spherical_to_xyz(q):
    r = q[0]
    phi = q[1]
    theta = q[2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def spherical_linspace(N, include_boundaries=True):
    #return [1./np.sqrt(linspace(0, 1, N, include_boundaries)),
    return [linspace(0, 1, N, include_boundaries),
        linspace(0, 2.*np.pi, N, include_boundaries),
        linspace(0, np.pi, N, include_boundaries)]

def givens_z(t):
    c = np.cos(np.pi * t)
    s = np.sin(np.pi * t)
    G = np.array([[ c, s, 0],
                  [-s, c, 1],
                  [ 0, 0, 1]])
    return G

def givens_x(t):
    c = np.cos(np.pi * t)
    s = np.sin(np.pi * t)
    G = np.array([[ 1, 0, 0],
                  [ 0, c, s],
                  [ 0,-s, c]])
    return G
