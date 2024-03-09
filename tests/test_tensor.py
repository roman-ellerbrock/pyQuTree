from qutree.ttn.tensor import *


def test_tensordot():
    # test that einsum and tensordot do the same thing
    t2 = np.arange(3*4).reshape((3, 4))
    t3 = np.arange(5*4*6).reshape((5, 4, 6))

    B_tensordot = np.tensordot(t2, t3, axes=([1], [1]))
    B_einsum = np.einsum(t2,[0, 1], t3, [2, 1, 3], [0, 2, 3])

    A2 = quTensor(t2, [(-3, 2), (2, 3)])
    A3 = quTensor(t3, [(3, 4), (2, 3), (1, 3)])
    B_qutree = tensordot(A2, A3, (2, 3))
    assert np.allclose(B_tensordot, B_einsum)
    assert np.allclose(B_tensordot, B_qutree)
    print('tensordot: ', B_tensordot)
    print('einsum:', B_einsum)
    print('qutree:', B_qutree)

def test_tensordot_mat():
    # test that einsum and tensordot do the same thing
    t2 = np.arange(4*4).reshape((4, 4))
    t3 = np.arange(5*4*6).reshape((5, 4, 6))

    B_tensordot = np.tensordot(t2, t3, axes=([1], [1]))
    B_einsum = np.einsum(t2, [0, 1], t3, [2, 1, 3], [0, 2, 3])

    A2 = quTensor(t2, [(2, 3), (2, 3)])
    A3 = quTensor(t3, [(3, 4), (2, 3), (1, 3)])
    B_qutree = tensordot(A2.transpose(), A3, (2, 3))
    assert np.allclose(B_tensordot, B_einsum)
    assert np.allclose(B_tensordot, B_qutree)
    print('tensordot: ', B_tensordot)
    print('einsum:', B_einsum)
    print('qutree:', B_qutree)
