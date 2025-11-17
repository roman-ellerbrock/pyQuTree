"""
Convenience functions for optimizing arbitrary Python functions.

This module provides a high-level interface for optimizing functions with
named parameters using tensor train or balanced tree networks.
"""
import numpy as np
from typing import Callable, Dict, Tuple, Optional, Union, List
from .ttn.ttnopt import Objective, ttnopt
from .ttn.network import tensor_train_graph, balanced_tree


class FunctionWrapper:
    """
    Wrapper to convert arbitrary functions with named parameters to array interface.

    This allows optimization of functions like:
        def my_func(x, y, z):
            return x**2 + y**2 + z**2

    Using the array-based TTNOpt interface.
    """

    def __init__(self, func: Callable, param_names: List[str]):
        """
        Initialize the function wrapper.

        Parameters
        ----------
        func : callable
            The function to optimize. Can accept named parameters or a dict.
        param_names : list of str
            Names of the parameters in order.
        """
        self.func = func
        self.param_names = param_names
        self.n_params = len(param_names)

    def __call__(self, x: np.ndarray) -> float:
        """
        Call the wrapped function with array input.

        Parameters
        ----------
        x : np.ndarray
            Array of parameter values.

        Returns
        -------
        float
            Function value.
        """
        # Try calling with keyword arguments first
        try:
            kwargs = {name: x[i] for i, name in enumerate(self.param_names)}
            return self.func(**kwargs)
        except TypeError:
            # If that fails, try calling with dict
            try:
                kwargs = {name: x[i] for i, name in enumerate(self.param_names)}
                return self.func(kwargs)
            except TypeError:
                # If that also fails, try calling with array directly
                return self.func(x)


def optimize_function(
    func: Callable,
    bounds: Dict[str, Tuple[float, float]],
    method: str = 'auto',
    n_sweeps: int = 3,
    bond_dim: int = 4,
    grid_points: Union[int, Dict[str, int]] = 21,
    start_points: Optional[Dict[str, List[float]]] = None,
    verbose: bool = True
) -> Dict:
    """
    Optimize an arbitrary function using tree tensor network optimization.

    This is a high-level convenience function that handles the conversion
    from arbitrary Python functions to the array-based TTNOpt interface.

    Parameters
    ----------
    func : callable
        Function to minimize. Can be:
        - Function with named parameters: func(x, y, z)
        - Function accepting dict: func({'x': 1, 'y': 2})
        - Function accepting array: func(np.array([1, 2, 3]))
    bounds : dict
        Dictionary mapping parameter names to (min, max) tuples.
        Example: {'x': (0, 10), 'y': (-5, 5), 'z': (0, 1)}
    method : str, default='auto'
        Optimization method:
        - 'auto': Choose based on dimensionality (TT for d<20, BT for d>=20)
        - 'tensor_train' or 'tt': Use tensor train network
        - 'balanced_tree' or 'bt': Use balanced tree network
    n_sweeps : int, default=3
        Number of optimization sweeps. More sweeps = better convergence.
    bond_dim : int, default=4
        Bond dimension r. Larger values = more accurate but slower.
    grid_points : int or dict, default=21
        Number of grid points per dimension. Can be:
        - int: Same number of points for all dimensions
        - dict: Per-parameter grid points, e.g. {'x': 21, 'y': 31, 'z': 11}
        More points = finer resolution but slower.
    start_points : dict, optional
        Initial guess points for warm-start. Format:
        {'x': [x1, x2, ...], 'y': [y1, y2, ...], ...}
        Each parameter should have >= bond_dim points.
    verbose : bool, default=True
        Print optimization progress and results.

    Returns
    -------
    dict
        Optimization results containing:
        - 'x': dict of optimal parameter values
        - 'fun': optimal function value
        - 'n_calls': number of function evaluations
        - 'n_cache_hits': number of cache hits
        - 'objective': Objective instance with full history
        - 'network': optimized tensor network

    Examples
    --------
    Optimize a simple quadratic function:

    >>> def rosenbrock(x, y):
    ...     return (1 - x)**2 + 100*(y - x**2)**2
    >>> bounds = {'x': (-2, 2), 'y': (-1, 3)}
    >>> result = optimize_function(rosenbrock, bounds)
    >>> print(f"Optimal x={result['x']['x']:.3f}, y={result['x']['y']:.3f}")
    >>> print(f"Minimum value: {result['fun']:.6f}")

    With per-parameter grid points:

    >>> # Use finer grid for y (narrow valley in Rosenbrock)
    >>> grid_pts = {'x': 21, 'y': 41}
    >>> result = optimize_function(rosenbrock, bounds, grid_points=grid_pts)

    With initial guess:

    >>> start = {'x': [0.5, 1.0, 1.5], 'y': [0.5, 1.0, 1.5]}
    >>> result = optimize_function(rosenbrock, bounds, start_points=start)

    High-dimensional problem:

    >>> def sphere(**kwargs):
    ...     return sum(v**2 for v in kwargs.values())
    >>> bounds = {f'x{i}': (-5, 5) for i in range(10)}
    >>> result = optimize_function(sphere, bounds, method='tensor_train')
    """
    # Extract parameter names and create ordered list
    param_names = list(bounds.keys())
    n_params = len(param_names)

    # Choose method automatically if needed
    if method == 'auto':
        method = 'balanced_tree' if n_params >= 20 else 'tensor_train'

    # Normalize method name
    if method in ['tensor_train', 'tt']:
        use_balanced_tree = False
    elif method in ['balanced_tree', 'bt']:
        use_balanced_tree = True
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'auto', 'tensor_train', or 'balanced_tree'.")

    # Handle grid_points: convert to dict if int, or validate dict
    if isinstance(grid_points, int):
        grid_points_dict = {param: grid_points for param in param_names}
    elif isinstance(grid_points, dict):
        # Validate all parameters have grid points specified
        for param in param_names:
            if param not in grid_points:
                raise ValueError(f"grid_points dict missing parameter '{param}'")
        grid_points_dict = grid_points
    else:
        raise ValueError("grid_points must be int or dict")

    if verbose:
        method_name = "Balanced Tree" if use_balanced_tree else "Tensor Train"
        print(f"Optimizing {n_params}D function using {method_name}")
        print(f"Parameters: {param_names}")
        if isinstance(grid_points, dict):
            grid_str = ", ".join(f"{p}:{grid_points_dict[p]}" for p in param_names)
            print(f"Settings: sweeps={n_sweeps}, bond_dim={bond_dim}, grid_points={{{grid_str}}}")
        else:
            print(f"Settings: sweeps={n_sweeps}, bond_dim={bond_dim}, grid_points={grid_points}")

    # Create wrapper for the function
    wrapper = FunctionWrapper(func, param_names)
    objective = Objective(wrapper)

    # Create primitive grids from bounds with per-parameter grid points
    primitive_grid = []
    for param in param_names:
        min_val, max_val = bounds[param]
        n_points = grid_points_dict[param]
        primitive_grid.append(np.linspace(min_val, max_val, num=n_points))

    # Create tensor network - pass primitive_grid so it knows the per-dimension sizes
    if use_balanced_tree:
        G = balanced_tree(n_params, bond_dim, primitive_grid)
    else:
        G = tensor_train_graph(n_params, bond_dim, primitive_grid)

    # Convert start_points if provided
    start_grid = None
    if start_points is not None:
        # Validate start_points
        for param in param_names:
            if param not in start_points:
                raise ValueError(f"start_points missing parameter '{param}'")
            if len(start_points[param]) < bond_dim:
                raise ValueError(
                    f"start_points['{param}'] has {len(start_points[param])} points, "
                    f"but need at least {bond_dim} (bond_dim)"
                )

        # Convert to array format (m x f) where m is number of points
        # Each row is a point, columns are dimensions
        start_lists = [start_points[param] for param in param_names]
        n_start_points = len(start_lists[0])
        start_grid = np.zeros((n_start_points, n_params))
        for i, param in enumerate(param_names):
            start_grid[:, i] = start_points[param]

    # Run optimization
    if verbose:
        print("Running optimization...")

    G_opt = ttnopt(G, objective, n_sweeps, primitive_grid, start_grid)

    # Extract results
    df = objective.logger.df
    best_idx = df['f'].idxmin()
    best_row = df.loc[best_idx]

    # Convert result to dict
    optimal_params = {}
    for i, param in enumerate(param_names):
        optimal_params[param] = best_row[f'x{i+1}']

    optimal_value = best_row['f']

    if verbose:
        print(f"\nOptimization complete!")
        print(f"Function evaluations: {objective.function_calls}")
        print(f"Cache hits: {objective.cache_hits}")
        print(f"Optimal value: {optimal_value:.6g}")
        print(f"Optimal parameters:")
        for param, value in optimal_params.items():
            print(f"  {param} = {value:.6g}")

    return {
        'x': optimal_params,
        'fun': optimal_value,
        'n_calls': objective.function_calls,
        'n_cache_hits': objective.cache_hits,
        'objective': objective,
        'network': G_opt,
    }


def minimize(
    func: Callable,
    bounds: Union[Dict[str, Tuple[float, float]], List[Tuple[float, float]]],
    **kwargs
) -> Dict:
    """
    Minimize a function using tree tensor network optimization.

    This is an alias for optimize_function with a scipy.optimize.minimize-like interface.

    Parameters
    ----------
    func : callable
        Function to minimize.
    bounds : dict or list of tuples
        Parameter bounds. If dict, maps parameter names to (min, max).
        If list, provides bounds for func(x) where x is array.
    **kwargs
        Additional arguments passed to optimize_function.

    Returns
    -------
    dict
        Optimization results (see optimize_function).

    See Also
    --------
    optimize_function : Full documentation of parameters and return values.
    """
    # Convert list of bounds to dict if needed
    if isinstance(bounds, (list, tuple)):
        bounds = {f'x{i}': b for i, b in enumerate(bounds)}

    return optimize_function(func, bounds, **kwargs)
