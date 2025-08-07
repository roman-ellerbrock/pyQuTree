'''
Optimization routines for grid-based tensor approximations in PyQuTree.

This module provides:
  - A base Model class defining the optimization API (sweep/optimize).
  - Utilities to evaluate grids of function values and select subsets via max-volume,
    greedy, or linear assignment methods (grid → matrix → grid).
  - Functions to generate candidate grids by "cross"-sampling one or multiple legs.
  - Two model implementations:
      * TensorRankOptimization: CP/Tensor-Rank optimizer.
      * MatrixTrainOptimization: General N-site matrix-train optimizer.
'''

import random
import itertools

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

    def sweep(self, grid: Grid, function: callable, epoch: int):
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

    def optimize(self, grid: Grid, function: callable, num_epochs: int) -> Grid:
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


def evaluate_grid(grid: Grid, function: callable, dim2: int, **kwargs) -> np.ndarray:
    """
    Evaluate `function` on each point in `grid`, then reshape to a (dim2 x dim1) matrix.

    Args:
        grid: Grid of shape (r*dim2 x f).
        function: Callable; grid.evaluate(V) returns flat array length r*dim2.
        dim2: Number of rows in the resulting matrix.

    Returns:
        mat:   Numpy array shape (dim2, dim1).
    """
    vmat = grid.evaluate(function, **kwargs)
    dim1 = int(vmat.size / dim2)
    return vmat.reshape(dim1, dim2).T


def random_points(primitive_grid: list[Grid], r: int) -> np.ndarray:
    """
    Sample r random points from each 1D primitive, returning an (r x f) array.
    """
    x = []
    for g in primitive_grid:
        pts = random.sample(list(g.grid.flatten()), r)
        x.append(pts)
    return np.array(x).T


def random_grid_points(primitive_grids: list[Grid], r: int) -> Grid:
    """
    Sample r unique points from the full Cartesian product of f primitives.

    Returns:
      Grid of shape (r x f) with coords [0..f-1].
    """
    def unique_integer_arrays(r, N, f):
        if r > N**f:
            raise ValueError("Not enough unique combos.")
        return np.array(random.sample(
            list(itertools.product(range(N), repeat=f)), r))

    def indices_to_grid_points(idxs, grids):
        return np.array([
            [grids[d].grid[i, 0] for d, i in enumerate(pt)]
            for pt in idxs
        ])

    f = len(primitive_grids)
    N = primitive_grids[0].grid.shape[0]
    idcs = unique_integer_arrays(r, N, f)
    coords = indices_to_grid_points(idcs, primitive_grids)
    return Grid(coords, list(range(f)))


def maxvol_selection(grid: Grid, function: callable, dim2: int, **kwargs):
    """
    Select rows of the (dim2 x dim1) evaluation matrix that maximize volume.

    Uses the `maxvol` algorithm.

    Returns:
        grid: Grid reduced to the selected rows.
        mat:  The full evaluation matrix before selection.
    """
    mat = evaluate_grid(grid, function, dim2, **kwargs)
    nidx, R = maxvol(mat)
    grid.grid = grid.grid[nidx, :]
    return grid, mat


def assignment_selection(grid: Grid, function: callable, dim2: int, **kwargs):
    """
    Select rows by solving a linear assignment problem on the cost matrix.

    Args:
        grid: Grid of candidate points.
        function: Objective callable.
        dim2: Number of rows in cost matrix.

    Returns:
        grid: Updated Grid with selected points.
        mat:  Cost matrix used for assignment.
    """
    mat = evaluate_grid(grid, function, dim2, **kwargs)
    rows, cols = linear_sum_assignment(mat)
    # map (row_i, col_j) back to flat index
    idcs = np.ravel_multi_index((rows, cols), mat.shape)
    grid.grid = grid.grid[idcs, :]
    return grid, mat


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


def greedy_selection(grid: Grid, function: callable, r: int, **kwargs):
    """
    Select grid points by choosing minimum in each column.

    Returns:
        grid: Updated Grid.
        vmat: Evaluation matrix used.
    """
    vmat = evaluate_grid(grid, function, r, **kwargs)
    rows, cols = greedy_column_min(vmat)
    idcs = np.ravel_multi_index((rows, cols), vmat.T.shape)
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


def greedy_with_group_assignment(matrix: np.ndarray,
                                 groups: np.ndarray) -> tuple[list[int], list[int]]:
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
        rows_g, cols_sub = linear_sum_assignment(matrix[:, cols_g])
        selected_rows[cols_g[cols_sub]] = rows_g
    return list(selected_rows), list(range(matrix.shape[1]))


def group_assignment(grid: Grid, function: callable,
                     groups: np.ndarray, r: int, **kwargs):
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


def variation_update(grid: Grid, replacement_grid: Grid,
                     function: callable, **kwargs) -> tuple[Grid, np.ndarray]:
    """
    One cross-update on a single physical leg:
      grid --create_mutations--> candidates
           --group_assignment--> new grid
    """
    ngrid, a = create_mutations(grid, replacement_grid)
    groups = column_labels(a.grid.T)
    return group_assignment(ngrid, function, groups, replacement_grid.num_points(), **kwargs)


def recombination_update(grid: Grid, idxs: list[int],
                         function: callable, **kwargs) -> tuple[Grid, np.ndarray]:
    """
    Recombine two subsets via max-volume selection.
    """
    ngrid = recombination(grid, idxs)
    return maxvol_selection(ngrid, function, grid.num_points(), **kwargs)


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

    def sweep(self, grid: Grid, function: callable, epoch: int):
        """
        One sweep: for each dimension k, perform `variation_update`.

        Returns:
          (new_grid, last_vmat)
        """
        for k in range(grid.num_coords()):
            grid, vmat = variation_update(
                grid,
                self.primitive_grids[k],
                function,
                epoch=epoch
            )
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

        self.cores = [
            random_grid_points(
                [primitive_grids[k],
                 primitive_grids[k+1],
                 primitive_grids[k+2]],
                r
            ) for k in range(self.N - 2)
        ]

    def data(self, primitive_grids: list[Grid], r: int):
        self.primitive_grids = primitive_grids
        self.r = r

    def sweep(self, _grid, function: callable, epoch: int):
        """
        One left-to-right pass.

        Returns:
          cores, vmat
        """
        vmat = None

        for k in range(self.N - 2):
            legs = [
                self.primitive_grids[k],
                self.primitive_grids[k+1],
                self.primitive_grids[k+2]
            ]
            self.cores[k], vmat = self._core_update(
                self.cores[k], legs, function, epoch=epoch,
            )
        return self.cores, vmat

    def _core_update(self, skel_core: Grid, leg_grids: list[Grid], function: callable, **kwargs) -> tuple[Grid, np.ndarray]:
        """
        Cross-update one triple-junction:
          1) create_mutations_multi on 3 legs
          2) evaluate -> flat array
          3) reshape to (Ni x (r*...))
          4) linear assignment select r rows
        Returns new_core and cost-matrix
        """
        candidates, _ = create_mutations_multi(skel_core, leg_grids)
        vals = candidates.evaluate(function, **kwargs)
        Ni = leg_grids[0].num_points()
        mat = vals.reshape(-1, Ni).T
        rows, cols = linear_sum_assignment(mat)
        idxs = np.ravel_multi_index((rows, cols), mat.shape)
        new_coords = candidates.grid[idxs, :]
        return Grid(new_coords, skel_core.coords), mat
