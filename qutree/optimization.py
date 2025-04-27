import numpy as np
import random
from qutree import cartesian_product
from qutree.matrix_factorizations.maxvol import maxvol
from scipy.optimize import linear_sum_assignment
import itertools
import random
from qutree.ttn.grid import Grid

class Model:
    def __init__(self):
        pass

    def sweep(self, grid, function, epoch):
        pass

    def optimize(self, grid, function, n_epoch):
        for epoch in range(n_epoch):
            grid = self.sweep(grid, function, epoch)
        return grid

def evaluate_grid(grid, V, dim2, **kwargs):
    v = grid.evaluate(V, **kwargs)
    dim1 = int(v.size / dim2)
    v = v.reshape(dim1, dim2).T
    return v

def maxvol_selection(grid, V, dim2, **kwargs):
    mat = evaluate_grid(grid, V, dim2, **kwargs)
    nidx, R = maxvol(mat)
    grid.grid = grid.grid[nidx, :]
    return grid, mat

def assignment_selection(grid, V, dim2, **kwargs):
    mat = evaluate_grid(grid, V, dim2, **kwargs)
    Veff = mat# - np.min(mat)
    rows, cols = linear_sum_assignment(Veff)
    idcs = np.ravel_multi_index((cols, rows), Veff.T.shape)
    grid.grid = grid.grid[idcs, :]
    return grid, mat

def rest(idx, grid):
    all = set(range(grid.coords.shape[0]))
    return sorted(all - set(idx))

def greedy_column_min(matrix):
    cols = matrix.argmin(axis=1)
    rows = list(range(matrix.shape[0])) # each row
    return rows, cols


def greedy_selection(grid, V, r, **kwargs):
    vmat = evaluate_grid(grid, V, r, **kwargs)
    rows, cols = greedy_column_min(vmat)
    idcs = np.ravel_multi_index((cols, rows), vmat.T.shape)
    grid.grid = grid.grid[idcs, :]
    return grid, vmat


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
    return c, a


def column_labels(matrix):
    matrix = np.array(matrix)
    _, labels = np.unique(matrix, axis=1, return_inverse=True)
    return labels


def greedy_with_group_assignment(matrix, groups):
    matrix = np.array(matrix)
    selected_rows = np.full(matrix.shape[1], -1)
    # selected_values = np.full(matrix.shape[1], np.nan)

    for g in set(groups):
        cols = np.flatnonzero(groups == g)
        r, c = linear_sum_assignment(matrix[:, cols])
        selected_rows[cols[c]] = r
        # selected_values[cols[c]] = matrix[r, cols[c]]

    # return selected_rows, selected_values
    return selected_rows, list(range(matrix.shape[1]))


def group_assignment(grid, V, groups, r, **kwargs):
    vmat = evaluate_grid(grid, V, r, **kwargs)
    # rows, cols = greedy_column_min(vmat)
    rows, cols = greedy_with_group_assignment(vmat, groups)
    idcs = np.ravel_multi_index((cols, rows), vmat.T.shape)
    grid.grid = grid.grid[idcs, :]
    return grid, vmat


def variation_update(grid, replacement_grid, V, r, **kwargs):
    ngrid, a = create_mutations(grid, replacement_grid)
    groups = column_labels(a.grid.T)
    return group_assignment(ngrid, V, groups, replacement_grid.num_points(), **kwargs)
    # return assignment_selection(ngrid, V, replacement_grid.num_points(), **kwargs)
    # return greedy_selection(ngrid, V, r, **kwargs)


def recombination_update(grid, idcs, V, r, **kwargs):
    ngrid = recombination(grid, idcs)
    return maxvol_selection(ngrid, V, grid.num_points(), **kwargs)
    # return greedy_selection(ngrid, V, r, **kwargs)

def random_points(primitive_grid, r):
    x = []
    for grid in primitive_grid:
        g = grid.grid.reshape(-1)

        subset = random.sample(list(g), r)
        x.append(subset)
    x = np.array(x).T
    return x


def random_grid_points(primitive_grids, r):
    def unique_integer_arrays(r, N, f):
        if r > N**f:
            raise ValueError("Not enough unique combinations for given length and range.")
        samples = random.sample(list(itertools.product(range(N), repeat=f)), r)
        return np.array(samples)

    def indices_to_grid_points(index_array, grid_linspaces):
        return np.array([[grid_linspaces[i].grid[idx][0] for i, idx in enumerate(point)]
                         for point in index_array])

    f = len(primitive_grids)
    N = primitive_grids[0].grid.shape[0]
    idcs = unique_integer_arrays(r, N, f)
    return Grid(indices_to_grid_points(idcs, primitive_grids), list(range(f)))


class TensorRankOptimization(Model):

    def __init__(self, primitive_grid, r):
        self.data(primitive_grid, r)

    def data(self, primitive_grid, r):
        self.primitive_grid = primitive_grid
        self.r = r

    def sweep(self, grid, function, epoch):
        for k in range(grid.num_coords()):
            grid, vmat = variation_update(grid, self.primitive_grid[k], function, self.r, epoch = epoch)

        return grid
