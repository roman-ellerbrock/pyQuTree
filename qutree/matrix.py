from .tensortrain import *


def permuteTensorizedMatrix(A: np.ndarray) -> np.ndarray:
    """
    permute a tensorized matrix:
    A[i1, i2, ..., in, j1, j2, ..., jn] -> A[i1, j1, i2, j2, ..., in, jn]
    [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 4, 1, 5, 2, 6, 3, 7]
    """
    dim = len(A.shape)
    x = np.arange(dim)
    x1, x2 = np.split(x, 2)
    perm = []
    for (i, j) in zip(x1, x2):
        perm.append(i)
        perm.append(j)
    A = np.transpose(A, perm)
    return A


def matrixToTensor(A: np.ndarray, permuteIndices: int = 1, leafDim = 2) -> np.ndarray:
    """
    * convert a matrix to a TT
    * A: matrix
    """
    # A[I, J] -> A[i1, i2, ..., in, j1, j2, ..., jn]
    dim1, dim2 = A.shape
    assert len(A.shape) == 2
    assert dim1 == dim2
    n = round(np.log2(dim1)/np.log2(leafDim))
    shape = tuple([leafDim]*2*n)
    A = A.reshape(shape)
    # A[i1, i2, ..., in, j1, j2, ..., jn] -> A[i1, j1, i2, j2, ..., in, jn]
    for i in range(permuteIndices):
        A = permuteTensorizedMatrix(A)
    return A


def compressMatrix(A: np.ndarray, rank: int, permuteIndices: int = 1, leafDim = 2) -> TensorTrain:
    B = matrixToTensor(A, permuteIndices, leafDim)
    return compress(B, rank)


def permuteTensorizedMatrixBack(A: np.ndarray) -> np.ndarray:
    """
    permute a tensorized matrix:
    A[i1, i2, ..., in, j1, j2, ..., jn] -> A[i1, j1, i2, j2, ..., in, jn]
    [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 2, 4, 6, 1, 3, 5, 7]
    """
    dim = len(A.shape)
    even = np.arange(0, dim, 2)
    odd = np.arange(1, dim, 2)
    perm = np.concatenate([even, odd])
    A = np.transpose(A, perm)
    return A


def tensorToMatrix(A: np.ndarray, permuteIndices: int = 1) -> np.ndarray:
    """
    * convert a Tensor to a matrix
    * A: Tensor
    """
    for i in range(permuteIndices):
        A = permuteTensorizedMatrixBack(A)
    shape = np.array(A.shape)
    shape1, shape2 = np.split(shape, 2)
    dim1 = np.prod(shape1)
    dim2 = np.prod(shape2)
    return A.reshape(dim1, dim2)


def decompressMatrix(tt: TensorTrain, permuteIndices: int = 1) -> np.ndarray:
    A = decompress(tt)
    return tensorToMatrix(A, permuteIndices)

def compress_decompress_matrix(A: np.ndarray, rank: int, permuteIndices: int = 1, leafDim = 2):
    tt = compressMatrix(A, rank, permuteIndices, leafDim)
    B = decompressMatrix(tt, permuteIndices)
    err = np.linalg.norm(A-B)
    return tt, err


