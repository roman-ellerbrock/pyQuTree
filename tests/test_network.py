from qutree import *

def test_start_sweep():
    tn = balanced_tree(4, 2, 10)
    #tn = tt_graph(4, 2, 10)
    plot_tree(tn)

    rt = root(tn)
    star = []
    star_sweep(tn, rt, None, star)
    print(tn.nodes)
    print(star)
    # expect = [(6, 5), (5, 1), (1, 5), (5, 0), (0, 5), (5, 6), (6, 4), (4, 3), (3, 4), (4, 2), (2, 4), (4, 6)]

def test_qutensor():
    e1 = [(-3, 2), (3, 2)]
    e2 = [(2, 3), (1, 3), (4, 3)]
    edge = (2, 3)
    At = np.arange(20*4).reshape((20, 4))
    Bt = np.arange(4*6*5).reshape((4, 6, 5))
    A = quTensor(At, e1)
    B = quTensor(Bt, e2)

    C2 = np.tensordot(At, Bt, axes=([1], [0]))
    C = tensordot(A, B, edge)

    assert np.allclose(C2, C)