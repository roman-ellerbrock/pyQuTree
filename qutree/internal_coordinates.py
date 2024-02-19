import numpy as np
from .ttn.grid import linspace

def spherical_to_xyz(q):
    r = q[0]
    theta = q[1]
    phi = q[2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def spherical_linspace(N, include_boundaries=True):
    return [linspace(0, 1, N, include_boundaries),
        linspace(0, 2.*np.pi, N, include_boundaries),
        linspace(0, np.pi, N, include_boundaries)]
