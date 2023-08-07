from qutree.sop import *


def test_a():
    psi = np.array([0., 1.])
    apsi = a() @ psi
    assert np.linalg.norm(apsi - [1., 0.]) < 1e-10

def test_adagger():
    psi = np.array([1., 0.])
    apsi = adagger() @ psi
    assert np.linalg.norm(apsi - [0., 1.]) < 1e-10

def test_n():
    n2 = np.array([[0., 0.], [0., 1.]])
    assert np.linalg.norm(n(), n2)