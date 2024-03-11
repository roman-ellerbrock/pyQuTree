from qutree import *

def test_balanced_tree():
    tn = balanced_tree(3, 2, 3)
    G = nx.DiGraph()
    G.add_edge(1, 3)
    G.add_edge(2, 3)
    G.add_edge(3, 4)
    for edge in G.edges():
        G.add_edge(*flip(edge))
    G.add_edge(-1, 4)
    G.add_edge(-2, 1)
    G.add_edge(-3, 2)

    for edge in G.edges():
        assert edge in tn.edges

def test_start_sweep():
    tn = balanced_tree(4, 2, 10)
    sweep = star_sweep(tn)
    expect = [(6, 5), (5, 1), (1, 5), (5, 0), (0, 5), (5, 6), (6, 4), (4, 3), (3, 4), (4, 2), (2, 4), (4, 6)]
    assert sweep == expect

def test_sweep():
    tn = balanced_tree(3, 2, 10)
#    expected = [(-3, 2), (-2, 1), (-1, 4), (1, 3), (2, 3), (3, 4), (4, 3), (3, 2), (3, 1)]
    expected = [(-2, 1), (-3, 2), (1, 3), (2, 3), (3, 4), (-1, 4), (4, -1), (4, 3), (3, 1), (3, 2), (1, -2), (2, -3)]
    assert sweep(tn) == expected

def test_pre_edges():
    G = balanced_tree(3, 2, 4)
    pre = sorted(pre_edges(G, (3, 4)))
    expect = sorted(((1, 3), (2, 3), (4, 3)))
    assert pre == expect
    pre = sorted(pre_edges(G, (3, 4), remove_flipped=True))
    expect = sorted(((1, 3), (2, 3)))
    assert pre == expect

def test_remove_edge():
    G = balanced_tree(3, 2, 4)
    remove_edge(G, (3, 4))
    assert (3, 4) not in G.edges
    assert (4, 3) not in G.edges
    assert 3 not in G.nodes
    expected = [(-3, 2), (-2, 1), (-1, 4), (1, -2), (1, 4), (2, -3), (2, 4), (4, -1)]
    assert sorted(G.edges) == expected

#def test_qutensor():
#    e1 = [(-3, 2), (3, 2)]
#    e2 = [(2, 3), (1, 3), (4, 3)]
#    edge = (2, 3)
#    At = np.arange(20*4).reshape((20, 4))
#    Bt = np.arange(4*6*5).reshape((4, 6, 5))
#    A = quTensor(At, e1)
#    B = quTensor(Bt, e2)
#
#    C2 = np.tensordot(At, Bt, axes=([1], [0]))
#    C = tensordot(A, B, edge)
#
#    assert np.allclose(C2, C)