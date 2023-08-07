import qutree.matrix as qt
import numpy as np


def test_compress_mat():
    A = np.arange(4**5)
    A = A.reshape([4, 4, 4, 4, 4])
    B = qt.compress(A, 4)
    assert len(B.tensors) == 5
    assert B.tensors[0].shape == (1, 4, 4)
    assert B.tensors[1].shape == (4, 4, 4)
    assert B.tensors[2].shape == (4, 4, 4)
    assert B.tensors[3].shape == (4, 4, 4)
    assert B.tensors[4].shape == (4, 4, 1)


def test_tt_to_tto():
    A = np.arange(4**5)
    A = A.reshape([4, 4, 4, 4, 4])
    B = qt.compress(A, 4)
    B = qt.tto(B)
    assert B.tensors[0].shape == (1, 2, 2, 4)
    assert B.tensors[1].shape == (4, 2, 2, 4)
    assert B.tensors[2].shape == (4, 2, 2, 4)
    assert B.tensors[3].shape == (4, 2, 2, 4)
    assert B.tensors[4].shape == (4, 2, 2, 1)


def test_permuteTensorizedMatrix():
    # create A(16, 15)
    A = np.arange(16*16)
    A = A.reshape(tuple([2] * 4 * 2))
    B = qt.permuteTensorizedMatrix(A)
    print(B.shape)
    C = qt.permuteTensorizedMatrixBack(B)
    assert np.linalg.norm(A-C) < 1e-12


def test_permuteTensorizedMatrix():
    # create A(16, 15)
    A = np.arange(16*16)
    A = A.reshape([16, 16])
    B = qt.matrixToTensor(A)
    assert B.shape == tuple([2] * 4 * 2)
    C = qt.tensorToMatrix(B)
    assert np.linalg.norm(A-C) < 1e-12


def test_compressMatrixArange():
    A = np.arange(16*16)
    A = A.reshape([16, 16])
    U, S, V = np.linalg.svd(A)
    B = qt.compressMatrix(A, 16)
    C = qt.decompressMatrix(B)
    assert np.linalg.norm(A-C) < 1e-10


def test_compressMatrixFFT():
    dim = 64
    A = np.fft.fftn(np.eye(dim, dim))
    B = qt.compressMatrix(A, 4)
    C = qt.decompressMatrix(B)
    assert np.linalg.norm(A-C) < 1e-11


def test_compressMatrixTridiagonal():
    dim = 64
    main_diag = np.random.rand(dim)
    upper_diag = np.random.rand(dim-1)
    lower_diag = np.random.rand(dim-1)
    A = np.diag(main_diag) + np.diag(upper_diag, k=1) + \
        np.diag(lower_diag, k=-1)

    B = qt.compressMatrix(A, 32)
    C = qt.decompressMatrix(B)
    assert np.linalg.norm(A-C) < 1e-11


def test_compressMatrixFFT():
    dim = 1024
    A = np.fft.fftn(np.eye(dim, dim))
    B = qt.compressMatrix(A, 6)
    C = qt.decompressMatrix(B)
    assert np.linalg.norm(A-C) < (1e-14*dim**2)

def test_compress_decompress_matrix():
    dim = 1024
    A = np.fft.fftn(np.eye(dim, dim))
    tt, err = qt.compress_decompress_matrix(A, 6)
    assert err < (1e-14*dim**2)

def test_compress_decompress_matrix_leafDim():
    dim = 1024
    leafdim = 4
    A = np.fft.fftn(np.eye(dim, dim))
    tt, err = qt.compress_decompress_matrix(A, 6, leafDim=leafdim)
    assert len(tt.tensors) == 10
    for t in tt.tensors:
        assert t.shape[1] == leafdim
    
    assert err < (1e-14*dim**2)