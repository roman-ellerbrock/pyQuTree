from qutree.ttn.tensor_network import *
from qutree.internal_coordinates import *

def test_contraction():
    N = 10
    r = 2
    f = 2
    # compress manually
    A0 = np.arange(N * r).reshape((N, r))
    A2 = np.arange(r*r).reshape((r, r))
    B = np.tensordot(A0, A2, axes=([1], [0]))
    A1 = np.arange(N * r).reshape((N, r))
    C = np.tensordot(B, A1, axes=([1], [1]))

    G = balanced_tree(f, r, N)
    G = create_tensors(G, np.arange, 'A')
    F = contract(G)
    x = F.nodes[2]['A']
    assert np.allclose(C, x)

def test_ttopt_contract1():
    def f_sin(x):
        y = x * np.pi
        return np.cos(y[0]) * np.sin(y[1])

    r = 2
    N = 20
    func = f_sin
    O = Objective(func, [linspace(-1, 1, N)] * 2)
    G = balanced_tree(2, r, N)
    G = tn_grid(G, O.linspace)
    Gopt, _ = ttnopt(G, O, nsweep = 5)

    F = contract(Gopt)
    F2 = F.nodes[2]['A']

    x1 = Grid(linspace(-1, 1, N), 0)
    x2 = Grid(linspace(-1, 1, N), 1)
    xyz = x1 @ x2
    Fref = xyz.evaluate(func)
    Fref = Fref.reshape((N, N))
    assert (np.linalg.norm(Fref - F2)/np.linalg.norm(Fref)) < 1e-12


def test_ttopt_contract2():
    def f_sin(x):
        y = x * np.pi
        return np.cos(y[0]) * np.sin(y[1])

    r = 2
    N = 20
    func = f_sin
    O = Objective(func, [linspace(-1, 1, N)] * 2)
    G = balanced_tree(2, r, N)
    G = tn_grid(G, O.linspace)
    Gopt, _ = ttnopt(G, O, nsweep = 5)

    F = extract_root_tensor(contract(Gopt))

    x1 = Grid(linspace(-1, 1, N), 0)
    x2 = Grid(linspace(-1, 1, N), 1)
    xyz = x1 @ x2
    Fref = xyz.evaluate(func)
    assert (np.linalg.norm(Fref - F)/np.linalg.norm(Fref)) < 1e-12

