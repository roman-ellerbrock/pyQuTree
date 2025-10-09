"""
Optimization routines for grid-based tensor approximations in PyQuTree.

This module provides:
  - A base Model class defining the optimization API (sweep/optimize).
  - Utilities to evaluate grids of function values and select subsets via max-volume,
    greedy, or linear assignment methods (grid → matrix → grid).
  - Functions to generate candidate grids by "cross"-sampling one or multiple legs.
  - Two model implementations:
      * TensorRankOptimization: CP/Tensor-Rank optimizer.
      * MatrixTrainOptimization: General N-site Matrix-Train optimizer.
"""

import random
import itertools
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from qutree.ttn.grid import Grid
from qutree import cartesian_product
from qutree.matrix_factorizations.maxvol import maxvol


class Model:
    """
    Base class for optimization models. Subclass must implement:
      - sweep(grid, function, epoch) -> (new_grid, aux_data)
      - optionally override optimize(...) to customize training loop.
    Provides a default optimize() that runs sweep() n_epoch times.
    """

    def __init__(self):
        pass

    def sweep(self, grid: Grid, function: Callable, epoch: int):
        """
        Perform one pass of the optimization.
        Should be implemented by subclasses.

        Args:
            grid:    Current skeleton/Grid of points.
            function: User-supplied scalar objective V(x) -> float.
            epoch:   Current epoch index (0-based).

        Returns:
            new_grid: Updated Grid.
            aux_data: Optional auxiliary output (e.g. evaluation matrix).
        """

    def optimize(self, grid: Grid, function: Callable, num_epochs: int) -> Grid:
        """
        Run `sweep` for `n_epochs` iterations starting from `grid`.

        Args:
            grid: Initial Grid.
            function: Objective callable.
            num_epochs: Number of sweeps to perform.

        Returns:
            Final updated Grid after n_epoch sweeps.
        """
        for epoch in range(num_epochs):
            grid, _ = self.sweep(grid, function, epoch)
        return grid


def evaluate_grid(grid: Grid, function: Callable, dim2: int, **kwargs) -> np.ndarray:
    """
    Evaluate `function` on each point in `grid`, then reshape to a (dim2 x dim1) matrix.

    Args:
        grid: Grid of shape (r*dim2 x f).
        function: Callable; grid.evaluate(V) returns flat array length r*dim2.
        dim2: Number of rows in the resulting matrix.

    Returns:
        vmat:   Numpy array shape (dim2, dim1).
    """
    vmat = grid.evaluate(function, **kwargs)
    dim1 = int(vmat.size / dim2)
    return vmat.reshape(dim1, dim2).T


def random_points(primitive_grid: list[Grid], r: int, seed:int=42) -> np.ndarray:
    """
    Sample r random points from each 1D primitive, returning an (r x f) array.
    """
    random.seed(seed)
    x = []
    for g in primitive_grid:
        pts = random.sample(list(g.grid.flatten()), r, seed=seed)
        x.append(pts)
    return np.array(x).T


# def random_grid_points(primitive_grids: list[Grid], r: int, seed:int=42) -> Grid:
#     """
#     Sample r unique points from the full Cartesian product of f primitives.

#     Returns:
#       Grid of shape (r x f) with coords [0..f-1].
#     """
#     random.seed(seed)
#     def unique_integer_arrays(r, N, f):
#         if r > N**f:
#             raise ValueError("Not enough unique combos.")
#         return np.array(random.sample(
#             list(itertools.product(range(N), repeat=f)), r))

#     def indices_to_grid_points(idxs, grids):
#         return np.array([
#             [grids[d].grid[i, 0] for d, i in enumerate(pt)]
#             for pt in idxs
#         ])

#     f = len(primitive_grids)
#     N = primitive_grids[0].grid.shape[0]
#     idcs = unique_integer_arrays(r, N, f)
#     coords = indices_to_grid_points(idcs, primitive_grids)
#     return Grid(coords, list(range(f)))

def random_grid_points(primitive_grids: list[Grid], r: int, seed: int = 42) -> Grid:
    random.seed(seed)
    sizes = [g.grid.shape[0] for g in primitive_grids]
    P = 1
    for n in sizes: P *= n
    if r > P:
        raise ValueError("Not enough unique combos.")

    # sample r unique linear indices without materializing the full product
    lin_idxs = random.sample(range(P), r)

    # convert linear indices to mixed-radix indices
    idcs = np.empty((r, len(sizes)), dtype=int)
    for k, v0 in enumerate(lin_idxs):
        v = v0
        for d in range(len(sizes) - 1, -1, -1):
            idcs[k, d] = v % sizes[d]
            v //= sizes[d]

    # map to grid coordinates
    coords = np.array([
        [primitive_grids[d].grid[i, 0] for d, i in enumerate(row)]
        for row in idcs
    ])
    return Grid(coords, list(range(len(sizes))))


def maxvol_selection(grid: Grid, function: Callable, dim2: int, **kwargs):
    """
    Select rows of the (dim2 x dim1) evaluation matrix that maximize volume.

    Uses the `maxvol` algorithm.

    Returns:
        grid: Grid reduced to the selected rows.
        vmat:  The full evaluation matrix before selection.
    """
    vmat = evaluate_grid(grid, function, dim2, **kwargs)
    nidx, R = maxvol(vmat)
    grid.grid = grid.grid[nidx, :]
    return grid, vmat


def assignment_selection(grid: Grid, function: Callable, dim2: int, **kwargs):
    """
    Select rows by solving a linear assignment problem on the cost matrix.

    Args:
        grid: Grid of candidate points.
        function: Objective callable.
        dim2: Number of rows in cost matrix.

    Returns:
        grid: Updated Grid with selected points.
        vmat:  Cost matrix used for assignment.
    """
    vmat = evaluate_grid(grid, function, dim2, **kwargs)
    rows, cols = linear_sum_assignment(vmat)
    # map (row_i, col_j) back to flat index
    idcs = np.ravel_multi_index((rows, cols), vmat.shape)
    grid.grid = grid.grid[idcs, :]
    return grid, vmat


def rest(idx: list[int], grid: Grid) -> list[int]:
    """
    Given a list of column indices `idx`, return the complement in [0..D-1].
    """
    all_cols = set(range(grid.coords.shape[0]))
    return sorted(all_cols - set(idx))


def greedy_column_min(matrix: np.ndarray) -> tuple[list[int], list[int]]:
    """
    Greedily select one minimum per column.

    Returns (rows, cols).
    """
    cols = matrix.argmin(axis=1)
    rows = list(range(matrix.shape[0]))
    return rows, cols


def greedy_selection(grid: Grid, function: Callable, r: int, **kwargs):
    """
    Select grid points by choosing minimum in each column.

    Returns:
        grid: Updated Grid.
        vmat: Evaluation matrix used.
    """
    vmat = evaluate_grid(grid, function, r, **kwargs)
    rows, cols = greedy_column_min(vmat)
    idcs = np.ravel_multi_index((rows, cols), vmat.shape)
    grid.grid = grid.grid[idcs, :]
    return grid, vmat


def recombination(grid: Grid, idxs: list[int]) -> Grid:
    """
    Recombine mutually exclusive grid components via cartesian product.

    Args:
        grid: Original Grid.
        idxs: Indices to keep.

    Returns:
        New permuted Grid of candidates.
    """
    a = grid[idxs]
    b = grid[rest(idxs, grid)]
    return cartesian_product([a, b]).permute()


def create_mutations(grid: Grid, replacement_grid: Grid) -> tuple[Grid, Grid]:
    """
    Cross-sample one leg: for each point in replacement_grid,
    hold the other coordinates fixed.

    Returns:
      candidates: Grid of shape (r*Nr x f).
      kept:       Grid of shape (r x f-1).
    """
    b = replacement_grid
    a = grid[rest(b.coords, grid)]
    c = cartesian_product([a, b]).permute()
    return c, a


def create_mutations_multi(skel: Grid, leg_grids: list[Grid]) -> tuple[Grid, Grid]:
    """
    Cross-sample all k legs simultaneously:
    - skel has shape (r x k)
    - leg_grids is list of k Grids (Ni x 1)

    Returns:
      candidates: Grid (r * ∏ Ni x k)
      kept:       Empty Grid (r x 0) as placeholder.
    """
    kept = Grid(np.zeros((skel.num_points(), 0)), np.array([], dtype=int))
    C = cartesian_product([kept] + leg_grids).permute()
    return C, kept


def column_labels(matrix: np.ndarray) -> np.ndarray:
    """
    Compute integer labels for unique columns of `matrix`.
    """
    _, labels = np.unique(matrix, axis=1, return_inverse=True)
    return labels


def greedy_with_group_assignment(
    matrix: np.ndarray,
    groups: np.ndarray
) -> tuple[list[int], list[int]]:
    """
    Run linear assignment poblem solver separately for each group of columns.

    Args:
      matrix: (R x C) cost matrix.
      groups: length-C array assigning each column to a group.

    Returns:
      rows:    selected row indices per column.
      cols:    column indices (0..C-1) in order.
    """
    matrix = np.array(matrix)
    selected_rows = np.full(matrix.shape[1], -1)
    for g in set(groups):
        cols_g = np.flatnonzero(groups == g)
        # @todo: first make maxvol inside a group submatrix and then do linear sum assignment inside the maxvol matrix
        rows_g, cols_sub = linear_sum_assignment(matrix[:, cols_g])
        selected_rows[cols_g[cols_sub]] = rows_g
    return list(selected_rows), list(range(matrix.shape[1]))


def group_assignment(
    grid: Grid,
    function: Callable,
    groups: np.ndarray,
    r: int,
    **kwargs
):
    """
    Grouped selection: select one row per column group via greedy_with_group_assignment.

    Returns:
      grid: Updated Grid.
      vmat: Evaluation matrix used.
    """
    vmat = evaluate_grid(grid, function, r, **kwargs)
    rows, cols = greedy_with_group_assignment(vmat, groups)
    idcs = np.ravel_multi_index((cols, rows), vmat.T.shape)
    grid.grid = grid.grid[idcs, :]
    return grid, vmat


def variation_update(
    grid: Grid,
    replacement_grid: Grid,
    function: Callable,
    **kwargs
) -> tuple[Grid, np.ndarray]:
    """
    One cross-update on a single physical leg:
      grid --create_mutations--> candidates
           --group_assignment--> new grid
    """
    ngrid, a = create_mutations(grid, replacement_grid)
    groups = column_labels(a.grid.T)
    return group_assignment(ngrid, function, groups, replacement_grid.num_points(), **kwargs)


def recombination_update(
    grid: Grid,
    idxs: list[int],
    function: Callable,
    **kwargs
) -> tuple[Grid, np.ndarray]:
    """
    Recombine two subsets via max-volume selection.
    """
    ngrid = recombination(grid, idxs)
    return maxvol_selection(ngrid, function, grid.num_points(), **kwargs)


def recombination_update_assignment(
    grid: Grid,
    left_block_cols: list[int],
    function: Callable,
    **kwargs
):
    """
    Recombine two blocks A (columns=left_block_cols) and B (the rest).
    - cartesian_product(A,B) -> candidate grid with r*r rows
    - evaluate -> r x r cost matrix (rows=r 'slots', cols=r candidates)
    - Hungarian assignment -> pick r rows for the next grid

    Returns:
      new_grid, vmat  with vmat.shape == (r, r)
    """
    ngrid = recombination(grid, left_block_cols)  # r^2 x f
    new_grid, vmat = assignment_selection(ngrid, function, dim2=grid.num_points(), **kwargs)
    return new_grid, vmat


class TensorRankOptimization(Model):
    """
    Tensor-Rank (PARAFAC) optimizer: performs one-leg cross-updates
    in a CP-format tree tensor network (TTNcross).

    Args:
      primitive_grids: list of f Grids, one per each dimension.
      r: Number of cross-pivots (rank).
    
    After a sweep, returns:
        grid: Updated Grid of shape (r x f).
        vmat: Evaluation matrix of shape (r x f).
    """

    def __init__(self, primitive_grids: list[Grid], r: int):
        self.data(primitive_grids, r)

    def data(self, primitive_grids: list[Grid], r: int):
        self.primitive_grids = primitive_grids
        self.r = r

    def sweep(self, grid: Grid, function: Callable, epoch: int):
        """
        One sweep: for each dimension k, perform `variation_update`.

        Returns:
          (new_grid, last_vmat)
        """
        vmat = None
        f = grid.num_coords()
        for k in range(f):
            grid, vmat = variation_update(grid, self.primitive_grids[k], function, epoch=epoch)
        return grid, vmat


class MatrixTrainOptimization(Model):
    """
    Optimizer for an N-site Matrix Train.

    Args:
        primitive_grids: list of N one-dimensional Grids (Ni x 1).
        r: Number of cross-pivots.

    After a sweep, returns:
      cores: list of N-2 Grid objects, each shape (r x 3).
      vmat : Evaluation matrix.
    """

    def __init__(self, primitive_grids: list[Grid], r: int):
        """
        Args:
          primitive_grids: list of N one-dimensional Grids (Ni x 1)
          r: skeleton junction rank
        """
        self.primitive_grids = primitive_grids
        self.N = len(primitive_grids)
        self.r = r

    def data(self, primitive_grids: list[Grid], r: int):
        self.primitive_grids = primitive_grids
        self.r = r

    def sweep(self, grid: Grid, function: Callable, epoch: int):
        vmat = None
        f = grid.num_coords()

        for k in range(min(2, f)):
            grid, _ = variation_update(grid, self.primitive_grids[k], function, epoch=epoch)

        for k in range(2, f-1):
            left_block = list(range(k))
            grid, vmat    = recombination_update_assignment(grid, left_block, function, epoch=epoch)
            grid, vmat = variation_update(grid, self.primitive_grids[k], function, epoch=epoch)

        if f>3:
            for k in range(f-1, f):
                grid, vmat = variation_update(grid, self.primitive_grids[k], function, epoch=epoch)

        return grid, vmat
