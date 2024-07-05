from qutree.matrix_factorizations.simultaneous_diagonalization import *
import numpy as np


def test_givens_rotation():
    n = 4
    U = np.eye(n)
    s, c = 1./np.sqrt(2.), 1./np.sqrt(2.)
    i, j = 2, 3
    U[i, i] = c
    U[i, j] = s
    U[j, i] = -np.conj(s)
    U[j, j] = c

    A = np.eye(n)
    Aref = U @ A @ U.T

    B = np.eye(n).reshape([1, n, n])
    givens_rotation(B, c, s, i, j)

    assert np.allclose(Aref, B)

def test_calc_angles():
    n = 2
    A = np.array([[0., 1.], [1., 0.]]).reshape([1, n, n])
    i, j = 0, 1
    c, s = calculate_angles(A, i, j)
    assert np.allclose(np.abs(c), 1./np.sqrt(2.))
    assert np.allclose(np.abs(s), 1./np.sqrt(2.))
    givens_rotation(A, c, s, i, j)
    assert np.allclose(A, np.array([[1., 0.], [0., -1.]]).reshape([1, n, n]))

def test_diagonalization1():
    A = np.array([[0., 1.], [1., 0.]]).reshape([1, 2, 2])
    A, U = simultaneous_diagonalization(A, n_iter=3)
    assert np.allclose(A, np.array([[[1., 0.], [0., -1.]]]))

def test_diagonalization2():
    n = 3
    A = np.arange(n * n).reshape([n, n])
    A = 0.5 * (A + A.T)
    eval, evec = np.linalg.eigh(A)
    A = A.reshape([1, n, n])
    eval_2, evec_2 = simultaneous_diagonalization(A.copy(), n_iter=50, verbose=False)
    A2 = evec_2 @ eval_2.reshape((n, n)) @ evec_2.T
    print('A  = ', A)
    print('A2 = ', A2)
    print('A3 = ', evec @ np.diag(eval) @ evec.T)

def test_diagonalization3():
    n = 5
    A = np.arange(n * n).reshape([n, n])
    A = 0.5 * (A + A.T)
    eval_1, evec = np.linalg.eigh(A)

    A = A.reshape([1, n, n])
    eval_2, evec_2 = simultaneous_diagonalization(A.copy(), n_iter=50, eps = 1e-15, verbose=False)
    assert off_diagonal_measure(eval_2) < 1e-5

    A2 = evec_2 @ eval_2.reshape((n, n)) @ evec_2.T
    assert np.allclose(A2, A)

    eval_2sort = np.array([eval_2[0, i, i] for i in range(n)])
    eval_2sort = np.sort(eval_2sort)
    assert np.allclose(eval_1, eval_2sort)

    assert np.allclose(evec_2 @ evec_2.T, np.eye(n))

#def test_diagonalization():
#    m, n = 1, 3
#    A = np.arange(n**2).reshape([m, n, n])
#    for i in range(m):
#        A[i] = 0.5 * (A[i] + A[i].T)
