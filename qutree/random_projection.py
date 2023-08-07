import numpy as np


def low_rank_svd(A: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(A)
    r = min(rank, U.shape[1])
    r = min(r, Vt.shape[0])
    U = U[:, :r]
    s = s[:r]
    Vt = Vt[:r, :]
    return U, s, Vt


def omega(rows: int, cols: int) -> np.ndarray:
    """
    Generate a random matrix of size n x m
    """
#    w = np.random.uniform(low = -1, high = 1., size=(rows, cols))
    w = np.random.normal(0., 1., size=(rows, cols))
    Q, R = np.linalg.qr(w)
    return Q


def svd(A: np.ndarray, rank: int, p: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD of a matrix A
    """
    M, N = A.shape
    n = min(rank, N, M)

    # precondition omega
    w = A @ omega(N, n)
    w, R = np.linalg.qr(w)
    for i in range(p):
        w = (A @ A.T) @ w
        w, R = np.linalg.qr(w)

    # compute randomized svd
    A = w.T @ A
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    U = w @ U

    return U, s, Vt
