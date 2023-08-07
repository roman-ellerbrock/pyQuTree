import numpy as np
from .random_projection import *
from sklearn.utils.extmath import randomized_svd


class TensorTrain:
    def __init__(self):
        self.tensors = []

    def append(self, A):
        if (isinstance(A, np.ndarray)):
            self.tensors.append(A)

    def __str__(self):
        return str(self.tensors)


def tensortrain(shape: tuple, gen, rank: int = 10):
    tt = TensorTrain()
    for i in range(len(shape)):
        if i == 0:
            A = gen(1 * shape[i] * rank)
            A = A.reshape([1, shape[i], rank])
            tt.tensors.append(A)
        elif i == len(shape)-1:
            A = gen([rank * shape[i] * 1])
            A = A.reshape([rank, shape[i], 1])
            tt.tensors.append(A)
        else:
            A = gen([rank * shape[i] * rank])
            A = A.reshape([rank, shape[i], rank])
            tt.tensors.append(A)
    return tt


def randTT(dof: int, rank: int, groupsize: int):
    tt = TensorTrain()
    for i in range(dof):
        if i == 0:
            tt.tensors.append(np.random.rand({1, 2**groupsize, rank}))
        elif i == dof-1:
            tt.tensors.append(np.random.rand({rank, 2**groupsize, 1}))
        else:
            tt.tensors.append(np.random.rand({rank, 2**groupsize, rank}))
    return tt


def compress(tensor: np.ndarray, rank: int) -> TensorTrain:
    """
    Compress a nD array into a TT
    A[i1, i2, ..., in] -> A1[i1m1] * A2[m1i2m2] *...* An[m(n-1)in]
    """
    shape = tensor.shape

    tt = TensorTrain()
    A = tensor
    A = np.expand_dims(A, axis=0)
    A = np.expand_dims(A, axis=-1)
    fidelity = 1.
    for i, dim in enumerate(shape):
        if i == len(shape) - 1:
            tt.tensors.append(A)
            break
        # reshape to matrix
        shapeA = A.shape
        dim1 = np.prod(A.shape[:2])
        dim2 = np.prod(A.shape[2:])
        A = A.reshape([dim1, dim2])

        # # contract
        r = min(rank, dim1, dim2)

#        U, s, Vt = low_rank_svd(A, r)
        U, s, Vt = svd(A, r, p=1)
#        U, s, Vt = randomized_svd(A, n_components=r, n_iter=0, random_state=0)

        A = np.diag(s) @ Vt

        # reshape U and save
        U = U.reshape(shapeA[0], shapeA[1], r)
        tt.tensors.append(U)

        # reshape A and continue
        A = A.reshape(r, *shapeA[2:])

    return tt


def decompress(tt: TensorTrain):
    A = None
    for i, T in enumerate(tt.tensors):
        if i == 0:
            A = T
        else:
            shapeA = A.shape
            shapeT = T.shape
            A = A.reshape(-1, shapeA[-1])
            T = T.reshape(shapeT[0], -1)
            A = A @ T
            A = A.reshape(*shapeA[:-1], *shapeT[1:])

    A = A.reshape(A.shape[1:-1])
    return A


def compress_decompress(A: np.ndarray, rank: int):
    tt = compress(A, rank)
    B = decompress(tt)
    err = np.linalg.norm(A-B)
    return tt, err


def tto(tt: TensorTrain):
    """
    * convert TT to TTO
    * tt: TT
    """
    for i in range(len(tt.tensors)):
        A = tt.tensors[i]
        dim = round(np.sqrt(A.shape[1]))
        tt.tensors[i] = A.reshape([A.shape[0], dim, dim, A.shape[2]])
    return tt
