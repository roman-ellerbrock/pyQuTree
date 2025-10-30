""" Benchmarking routines for optimization algorithms. """

from typing import Callable, List, Tuple
import random as pyrandom
import numpy as np
from scipy.optimize import differential_evolution, dual_annealing

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


def _dim(bounds: List[Tuple[float, float]]) -> int:
    """Get problem dimension from bounds."""
    return len(bounds)


def _budget(num_grid_points: int, rank: int, num_sweeps: int, D: int) -> int:
    """Compute budget (number of function calls) for baseline SciPy optimizers."""
    return max(100, rank * num_grid_points * num_sweeps * D)


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


def run_de(func: Callable, bounds: List[Tuple[float, float]],
           num_grid_points: int, rank: int, num_sweeps: int, seed: int = 42):
    """SciPy Differential Evolution baseline (global). Returns (calls, best_f_raw, obj)."""
    _seed_all(seed)
    num_dims = _dim(bounds)
    budget = _budget(num_grid_points, rank, num_sweeps, num_dims)

    # Use same transform as TRC/MT: score t = -exp(-f)
    obj = Objective(func, lambda x: -np.exp(-x))

    calls = 0
    best_fx = np.inf

    def f(x):
        nonlocal calls, best_fx
        calls += 1
        t = obj(np.asarray(x))  # logs score and increments obj.function_calls
        # Invert t = -exp(-f) -> f = -ln(-t)
        if t < 0:
            fx = float(-np.log(np.clip(-t, 1e-100, 1.0)))
            if fx < best_fx:
                best_fx = fx
        return t

    popsize = 10
    maxiter = max(1, budget // max(popsize * D, 1))
    _ = differential_evolution(
        f, bounds, seed=seed, popsize=popsize, maxiter=maxiter,
        polish=False, updating="deferred", workers=1, tol=0.0
    )

    total_calls = getattr(obj, "function_calls", calls)
    return total_calls, best_fx, obj


def run_da(func: Callable, bounds: List[Tuple[float, float]],
           num_grid_points: int, rank: int, num_sweeps: int, seed: int = 42):
    """SciPy Dual Annealing baseline (global SA). Returns (calls, best_f_raw, obj)."""
    _seed_all(seed)
    num_dims = _dim(bounds)
    budget = _budget(num_grid_points, rank, num_sweeps, num_dims)

    obj = Objective(func, lambda x: -np.exp(-x))

    calls = 0
    best_fx = np.inf

    def f(x):
        nonlocal calls, best_fx
        calls += 1
        t = obj(np.asarray(x))
        if t < 0:
            fx = float(-np.log(np.clip(-t, 1e-100, 1.0)))
            if fx < best_fx:
                best_fx = fx
        return t

    _ = dual_annealing(f, bounds=bounds, seed=seed, maxfun=budget, no_local_search=True)

    total_calls = getattr(obj, "function_calls", calls)
    return total_calls, best_fx, obj
