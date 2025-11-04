import numpy as np
from .tensor import *
from .network import *
from ..matrix_factorizations.maxvol import maxvol

def _cartesian_product(A, B):
    return np.array([[*a, *b] for a in A for b in B])

class Grid:
    def __init__(self, grid, coords):
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid)
        if len(grid.shape) == 1:
            grid = grid.reshape(-1, 1)
        if len(grid.shape) != 2:
            raise ValueError("Grid must be a 2D array.")
        self.grid = np.array(grid)

        if isinstance(coords, int):
            coords = [coords]
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)
        self.coords = coords

        if grid.shape[1] != coords.shape[0]:
            raise ValueError("Number of columns in grid must match the number of rows in coords.")

        self.permutation = np.argsort(self.coords)

    def __matmul__(self, other):
        new_grid = _cartesian_product(self.grid, other.grid)
        new_coords = np.concatenate((self.coords, other.coords))
        return Grid(new_grid, new_coords)

    def permute(self):
        new_grid = self.grid[:, self.permutation]
        new_coords = self.coords[self.permutation]
        G = Grid(new_grid, new_coords)
        G.permutation = np.argsort(self.permutation)
        return G

    def __str__(self):
        output = "coords: {}\ngrid:\n{}".format(self.coords, self.grid)
        return output

    def random_subset(self, n):
        m = min(n, self.grid.shape[0])
        idx = np.random.choice(self.grid.shape[0], m, replace=False)
        return Grid(self.grid[idx], self.coords)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise ValueError("Index must be a tuple of two integers.")
            grid = self.grid[idx[0], idx[1]]
            coords = self.coords[idx[1]]
            return Grid(grid, coords)
        if not isinstance(idx, int):
            grid = self.grid[:, idx]
            coords = self.coords[idx]
            return Grid(grid, coords)
        grid = self.grid[idx]
        return Grid(grid, self.coords)
    
    def evaluate(self, func, **kwargs):
        return np.apply_along_axis(func, 1, self.grid, **kwargs)
    
    def transform(self, func):
        return Grid(np.apply_along_axis(func, 1, self.grid), self.coords)
    
    def __add__(self, other):
        if len(self.coords) != len(other.coords):
            raise ValueError("Number of coordinates does not match.")
        if self.coords.all() != other.coords.all():
            raise ValueError("Coordinates do not match: ", self.coords, " | ", other.coords, ".")
        return Grid(np.concatenate((self.grid, other.grid), axis=0), self.coords)
    
    def shape(self):
        return self.grid.shape
    
    def num_points(self):
        return self.shape()[0]
    
    def num_coords(self):
        return self.shape()[1]

def cartesian_product(grids):
    if len(grids) == 1:
        return grids[0]
    return grids[0] @ cartesian_product(grids[1:])

def direct_sum(grids):
    if len(grids) == 1:
        return grids[0]
    return grids[0] + direct_sum(grids[1:])

def build_node_grid(G):
    for node in G.nodes:
        if is_leaf_node(node, G):
            continue
        edges = G.in_edges(node)
        pre_grids = collect(G, edges, 'grid')
        grid = cartesian_product(pre_grids).permute()
        G.nodes[node]['grid'] = grid

def tn_grid(G, primitive_grid, start_grid = None):
    """
    Initialize a random tn grid
    G: tensor network
    grids: list of grids for each coordinate
    """

    for leaf in sorted(up_leaves(G)):
        coord = G.edges[leaf]['coordinate']
        G.edges[leaf]['grid'] = Grid(primitive_grid[coord], coord)

    for edge in sweep(G, False):
        r = G.edges[edge]['r']
        pre = pre_edges(G, edge, remove_flipped=True)
        pre_grids = collect(G, pre, 'grid')
        next_grid = cartesian_product(pre_grids).random_subset(r)
        if start_grid is not None:
            assert start_grid.shape[0] >= r
            next_grid.grid = start_grid[:r, next_grid.coords]
        G.edges[edge]['grid'] = next_grid

    build_node_grid(G)
    return G

def transform_node_grid(G, q_to_x):
    for node in G.nodes:
        if node < 0:
            continue
        G.nodes[node]['grid'] = G.nodes[node]['grid'].transform(q_to_x)
    return G

def regularized_inverse(A: np.ndarray, lambda_reg: float, eps: float = 1e-15) -> np.ndarray:
    """
    Tikhonov-regularized inverse via SVD:
      sigma_inv = s / (s^2 + alpha),  with  alpha = (lambda_eff * s_max)^2
    where lambda_eff = max(lambda_reg, eps). Denominator is clamped to >= eps.
    This prevents divide-by-zero/overflow and stays stable even if s_max <= 0.
    """
    U, sigma, VT = np.linalg.svd(A, full_matrices=False)
    # Ensure float64 for stability
    U = U.astype(np.float64, copy=False)
    sigma = sigma.astype(np.float64, copy=False)
    VT = VT.astype(np.float64, copy=False)

    if sigma.size == 0:
        # Empty spectrum: return shape-consistent zero inverse
        return np.zeros_like(A.T, dtype=np.float64)

    s_max = float(np.max(sigma))  # safer than sigma[0] if SVD ordering ever changes
    # Enforce strictly positive regularization
    lam = float(lambda_reg)
    if not np.isfinite(lam) or lam <= 0.0:
        lam = eps

    # If s_max is non-positive or non-finite, still produce a positive alpha
    if not np.isfinite(s_max) or s_max <= 0.0:
        alpha = lam * lam
    else:
        alpha = (lam * s_max) ** 2

    # Denominator, safely clamped away from zero
    denom = sigma * sigma + alpha
    denom = np.where(denom <= eps, eps, denom)

    sigma_inv = sigma / denom  # guaranteed finite

    # A_inv = V * diag(sigma_inv) * U^T
    return (VT.T * sigma_inv) @ U.T

def maxvol_grids(A, G, edge):
    pre = pre_edges(G, edge, remove_flipped=True)
    grid_L = collect(G, pre, 'grid')
    grid_L = cartesian_product(grid_L)

    mat = A.flatten(edge)

    rows, _ = maxvol(mat)

    # compute cross matrix inverse
    cross_inv = regularized_inverse(mat[rows, :], 1e-12)
    cross_inv = quTensor(cross_inv, [edge, flip(edge)])
    return grid_L[rows, :], cross_inv
