""" Benchmarking routines for optimization algorithms. """

import random as pyrandom
import numpy as np

from qutree import (
    Grid,
    Objective,
    tensor_train_graph,
    ttnopt,
)
from qutree.optimization import (
    TensorRankOptimization,
    MatrixTrainOptimization,
    random_grid_points,
)


def _seed_all(seed):
    """Seed all random number generators for reproducibility."""
    np.random.seed(seed)
    pyrandom.seed(seed)


def make_primitives(bounds, num_grid_points):
    """Create list of 1D grids (primitives) for given bounds."""
    return [
        Grid(
            np.linspace(lo, hi, num_grid_points, endpoint=True), [k]
        ) for k, (lo, hi) in enumerate(bounds)
    ]

def run_trc(func, bounds, num_grid_points, rank, num_sweeps, seed=42):
    """Run Tensor Rank Cross benchmark."""
    _seed_all(seed)
    primitives = make_primitives(bounds, num_grid_points)
    #obj = Objective(func)
    obj = Objective(func, lambda x: -np.exp(-x))
    model = TensorRankOptimization(primitives, rank)
    grid_start = random_grid_points(primitives, rank, seed)
    _ = model.optimize(grid_start, obj, num_sweeps)
    calls = obj.function_calls
    f_min = float(np.min(obj.logger.df['f'])) if len(obj.logger.df) else np.nan
    return calls, f_min, obj

def run_mt(func, bounds, num_grid_points, rank, num_sweeps, seed=42):
    """Run Matrix Train benchmark."""
    _seed_all(seed)
    primitives = make_primitives(bounds, num_grid_points=num_grid_points)
    #obj = Objective(func)
    obj = Objective(func, lambda x: -np.exp(-x))
    model = MatrixTrainOptimization(primitives, rank)
    grid_start = random_grid_points(primitives, rank, seed)
    _ = model.optimize(grid_start, obj, num_sweeps)
    calls = obj.function_calls
    f_min = float(np.min(obj.logger.df['f'])) if len(obj.logger.df) else np.nan
    return calls, f_min, obj

def run_ttopt(func, bounds, num_grid_points, rank, num_sweeps, seed=42):
    """Run Tensor Train Optimization benchmark."""
    _seed_all(seed)
    f = len(bounds)
    #obj = Objective(func)
    obj = Objective(func, lambda x: np.exp(-x))
    ttgraph = tensor_train_graph(f, rank, num_grid_points)
    primitive_grid = [np.linspace(lo, hi, num_grid_points, endpoint=True) for (lo, hi) in bounds]
    _ = ttnopt(ttgraph, obj, num_sweeps, primitive_grid)
    calls = obj.function_calls
    f_min = float(np.min(obj.logger.df['f'])) if len(obj.logger.df) else np.nan
    return calls, f_min, obj
