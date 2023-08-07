import qutree as qt
import numpy as np
import copy


def test_low_rank_svd():
    N = 16
    rank = 4
    A = np.random.rand(N*N).reshape([N, N])
    U, s, Vt = qt.low_rank_svd(A, rank)
    assert U.shape == (N, rank)
    assert s.shape == (rank,)
    assert Vt.shape == (rank, N)


def test_svd():
    N = 16
    rank = 4
    A = np.random.rand(N*N).reshape([N, N])

    U0, s0, Vt0 = qt.low_rank_svd(A, rank)
    A = U0 @ np.diag(s0) @ Vt0

    U, s, Vt = qt.svd(A, rank)
    B = U @ np.diag(s) @ Vt

    assert np.linalg.norm(B - A) < 1e-11
    assert U.shape == (N, rank)
    assert s.shape == (rank,)
    assert Vt.shape == (rank, N)

