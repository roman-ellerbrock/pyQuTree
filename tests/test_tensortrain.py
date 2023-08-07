import qutree as qt
import numpy as np
import unittest


def test_compress():
    A = np.ndarray([2, 2, 2, 2, 2])
    B = qt.compress(A, 4)
    assert len(B.tensors) == 5
    assert B.tensors[0].shape == (1, 2, 2)
    assert B.tensors[1].shape == (2, 2, 4)
    assert B.tensors[2].shape == (4, 2, 4)
    assert B.tensors[3].shape == (4, 2, 2)
    assert B.tensors[4].shape == (2, 2, 1)


def test_compress_decompress():
    n = 6
    rank = 5
    A = np.arange(2**n)
    A = A.reshape([2] * n)
    B, err = qt.compress_decompress(A, rank)
    assert (err < 1e-12)


def test_tensortrain():
    shape = tuple([2] * 20)
    dim = np.prod(shape)
    rank = 20

    A = np.arange(np.prod(shape))
    A = A.reshape(shape)
    A = A/np.max(A)

    tt, err = qt.compress_decompress(A, rank)
    assert (err < 1e-15*dim)
