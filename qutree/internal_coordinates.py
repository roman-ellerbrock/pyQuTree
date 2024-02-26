import numpy as np

def linspace(xa = 0., xe = 1., N = 1, include_boundaries = False):
    """
    Returns equidistant grid with N points between xa and xe.
    Boundary (xa, xe) can be included (include_boundaries = True).
    Note that, for example, for integration with trapezoidal rule,
    the boundaries should be excluded.
    """
    if include_boundaries:
        return np.linspace(xa, xe, N)
    else:
        return np.linspace(xa + (xe - xa) / (2*N), xe - (xe - xa) / (2*N), N)

def linspace_ddx(xa = 0., xe = 1., N = 1, include_boundaries = False):
    # qutree:
    # dp = 1. / (xe - xa)
    # L = N * dp
    # here:
    # dx = (xe - xa) / N
    # dp = 2pi / dx = 2pi / (xe - xa) 
    L = 2. * np.pi * N / (xe - xa) # the more grid points, the higher possible p
    p = linspace(-L / 2., L / 2., N, include_boundaries)
    return p

def apply_p(x, f_x, power = 1):
    # Note: recommended to use ddx_stencil instead!
    # Compute the DFT of f(x)
    F_p = np.fft.fft(f_x)
    # Define the wavenumbers
    N = len(x)
    k = 2 * np.pi * N / (x[-1] - x[0]) * np.fft.fftfreq(N)
    # Multiply F_p by k
    F_p *= (k * 1j)**power
    # Compute the IDFT to obtain the result in x-space
    result_x = np.fft.ifft(F_p)
    return result_x

def ddx_stencil(f, x, power = 1):
    dx = x[1] - x[0]
    if power == 1:
        dfdx = 0.5 * (np.roll(f, -1) - np.roll(f, 1)) / (dx)
        return dfdx
    elif power == 2:
        dfdx = (np.roll(f, -1) - 2*f + np.roll(f, 1)) / (dx**2)
        return dfdx
    else:
        raise ValueError("power must be 1 or 2")


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
