import json
import numpy as np

class GTO:
    def __init__(self):
        self.exp = np.zeros(6)
        self.coeff = np.zeros(6)
        self.origin = np.zeros(3)
        self.lmn = [0, 0, 0]

    def __call__(self, x: np.ndarray):
        val = 0.
        x1 = x - self.origin
        for c, e in zip(self.coeff, self.exp):
            val += c * (x1[0]**self.lmn[0])*np.exp(-e*np.linalg.norm(x1)**2)
#            val += c * (x1[0]**self.lmn[0])*(x1[1]**self.lmn[1])*(x1[2]**self.lmn[2])*np.exp(-e*np.linalg.norm(x1)**2)
        return val

def readBasis(name: str):
    with open(name, 'r') as file:
        data = json.load(file)
        return data

def gto_direct(exps, coeff, origin: np.ndarray, lmn: list):
    g = GTO()
    g.exp = np.array(exps, dtype='float64')
    g.coeff = np.array(coeff, dtype='float64')
    g.origin = np.array(origin, dtype='float64')
    g.lmn = lmn
    return g

def gto(basis, element: int, shell: int, lmn: list, origin: np.ndarray):
    sh = basis["elements"][str(element)]["electron_shells"][shell]

    coeff = sh["coefficients"]
    exp = sh["exponents"]
    ls = np.array(sh["angular_momentum"])
    l = np.sum(lmn)
    indices = np.where(ls == l)

    if len(indices) == 0:
        raise 'Invalid angular momentum.'
    if len(indices[0]) == 0:
        raise 'Invalid angular momentum (v2).'

    idx = indices[0][0]
    return gto_direct(exp, coeff[idx], origin, lmn)

def eval(gto, X):
    N = len(X)
    f = np.ndarray([N])
    for I, x in enumerate(X):
        f[I] = gto(x)
    return f




