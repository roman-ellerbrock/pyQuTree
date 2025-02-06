import numpy as np

from qutree import cartesian_product
from qutree.matrix_factorizations.maxvol import maxvol

class Model:
    def __init__(self):
        pass

    def sweep(self, grid, function, epoch):
        pass

    def optimize(self, grid, function, n_epoch):
        for epoch in range(n_epoch):
            grid = self.sweep(grid, function, epoch)
        return grid

def maxvol_selection(grid, V, dim2, **kwargs):
    v = grid.evaluate(V, **kwargs)
    v = v.reshape((int(v.size / dim2), dim2)).T
    nidx, R = maxvol(v)
    grid.grid = grid.grid[nidx, :]
    return grid

def rest(idx, grid):
    all = set(range(grid.coords.shape[0]))
    return sorted(all - set(idx))


def greedy_selection(grid, V, r, **kwargs):
    # grid = grid.copy()
    vmat = grid.evaluate(V, **kwargs)
    # for i in range(vmat.shape[0]):
    #     print(i, grid.grid[i, :], vmat[i])
    # print(np.argsort(vmat))
    nidx = np.argsort(vmat)[:r]
    # nidx = np.argsort(vmat)[-r:]
    grid.grid = grid.grid[nidx, :]
    return grid


def recombination(grid, idxs):
    """
    recombine mutually exclusive grid components
    """
    a = grid[idxs]
    b = grid[rest(idxs, grid)]
    return cartesian_product([a, b]).permute()


def create_mutations(grid, replacement_grid):
    b = replacement_grid
    a = grid[rest(b.coords, grid)]
    c = cartesian_product([a, b]).permute()
    return c


def variation_update(grid, replacement_grid, V, r, **kwargs):
    ngrid = create_mutations(grid, replacement_grid)
    return maxvol_selection(ngrid, V, replacement_grid.num_points(), **kwargs)
    # return greedy_selection(ngrid, V, r, **kwargs)


def recombination_update(grid, idcs, V, r, **kwargs):
    ngrid = recombination(grid, idcs)
    return maxvol_selection(ngrid, V, grid.num_points(), **kwargs)

