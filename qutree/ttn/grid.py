import numpy as np
from .network import *
from functools import partial
from ..matrix_factorizations.maxvol import maxvol

def linspace(xa = 0., xe = 0., N = 1, include_boundaries = False):
    """
    Returns equidistant grid with N points between xa and xe.
    Boundary (xa, xe) can be included (include_boundaries = True).
    Note that, for example, for integration with trapezoidal rule,
    the boundaries should be excluded.
    """
    if include_boundaries:
        return np.linspace(xa, xe, N)
    else:
        return np.linspace(xa + 0.5 / N, xe - 0.5 / N, N)

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
        if n > self.grid.shape[0]:
            raise ValueError("Subset size cannot be larger than the grid size.")
        idx = np.random.choice(self.grid.shape[0], n, replace=False)
        return Grid(self.grid[idx], self.coords)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise ValueError("Index must be a tuple of two integers.")
            grid = self.grid[idx[0], idx[1]]
            coords = self.coords[idx[1]]
            return Grid(grid, coords)
        if not isinstance(idx, int):
            raise ValueError("Index must be an integer or a tuple of two integers.")
        grid = self.grid[idx]
        return Grid(grid, self.coords)
    
    def evaluate(self, func):
        return np.apply_along_axis(func, 1, self.grid)
    
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
        if node < 0:
            continue
        edges = G.in_edges(node)
        pre_grids = collect(G, edges, 'grid')
        grid = cartesian_product(pre_grids).permute()
        G.nodes[node]['grid'] = grid

def tn_grid(G, grids):
    """
    Initialize a random tn grid
    G: tensor network
    grids: list of grids for each coordinate
    """
    for edge in sweep(G):
        if (is_leaf(edge)):
            coord = get_coordinate(edge)
            G[edge[0]][edge[1]]['grid'] = Grid(grids[coord], coord)
        else:
            r = G.edges[edge]['r']
            pre = pre_edges(G, edge, remove_flipped=True)
            pre_grids = collect(G, pre, 'grid')
            next_grid = cartesian_product(pre_grids).random_subset(r)
            G[edge[0]][edge[1]]['grid'] = next_grid

    build_node_grid(G)
    return G

def transform_node_grid(G, q_to_x):
    for node in G.nodes:
        if node < 0:
            continue
        G.nodes[node]['grid'] = G.nodes[node]['grid'].transform(q_to_x)
    return G

def maxvol_grids(grids, f):
    grid_L = cartesian_product(grids[:-1])
    grid_R = grids[-1]
    grid = grid_L @ grid_R
    r = grid_R.shape()[0]

    mat = grid.evaluate(f)
    mat = mat.reshape([-1, r])
    if mat.shape[0] == mat.shape[1]:
        return grid_L
    rows, _ = maxvol(mat)
    return grid_L[rows, :]
