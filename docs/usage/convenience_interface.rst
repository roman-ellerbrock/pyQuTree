Convenience Interface
=====================

The convenience interface provides a simple, high-level API for optimizing arbitrary Python functions without needing to understand the details of tensor networks.

Quick Start
-----------

Basic optimization of a function with named parameters:

.. code-block:: python

    from qutree import optimize_function

    # Define your function with named parameters
    def rosenbrock(x, y):
        return (1 - x)**2 + 100*(y - x**2)**2

    # Define parameter bounds
    bounds = {'x': (-2, 2), 'y': (-1, 3)}

    # Optimize!
    result = optimize_function(rosenbrock, bounds)
    print(f"Optimal x={result['x']['x']:.3f}, y={result['x']['y']:.3f}")
    print(f"Minimum value: {result['fun']:.6f}")

Function Signatures
-------------------

The convenience interface supports multiple function signatures:

**Named Parameters**

.. code-block:: python

    def my_func(x, y, z):
        return x**2 + y**2 + z**2

    bounds = {'x': (-5, 5), 'y': (-5, 5), 'z': (-5, 5)}
    result = optimize_function(my_func, bounds)

**Dictionary Parameter**

.. code-block:: python

    def my_func(params):
        return params['x']**2 + params['y']**2

    bounds = {'x': (-5, 5), 'y': (-5, 5)}
    result = optimize_function(my_func, bounds)

**Array Parameter**

.. code-block:: python

    def my_func(x):
        return np.sum(x**2)

    # Can use list of bounds for array functions
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    result = minimize(my_func, bounds)

Per-Parameter Grid Points
--------------------------

Use different grid resolutions for different parameters to balance accuracy and computational cost:

.. code-block:: python

    def anisotropic(x, y, z):
        # y has a narrow valley, needs finer resolution
        return (x - 1)**2 + 100*(y - 0.5)**2 + (z + 1)**2

    bounds = {'x': (-2, 3), 'y': (-1, 2), 'z': (-3, 1)}

    # Use different grid resolutions
    grid_points = {
        'x': 11,  # Coarse grid
        'y': 31,  # Fine grid for narrow valley
        'z': 11   # Coarse grid
    }

    result = optimize_function(
        anisotropic,
        bounds,
        grid_points=grid_points,
        n_sweeps=3
    )

**When to use per-parameter grids:**

- Functions with narrow valleys in some dimensions
- Parameters with different sensitivities
- High-dimensional problems where you want to focus resolution

This can significantly reduce computational cost. In the example above, using per-parameter grids requires ~3,700 evaluations vs ~26,000 for a uniform 31×31×31 grid.

Warm-Start Optimization
-----------------------

Provide initial guess points to speed up convergence:

.. code-block:: python

    def quadratic(x, y):
        return (x - 1.5)**2 + (y + 0.5)**2

    bounds = {'x': (0, 3), 'y': (-2, 1)}

    # Provide start points near the optimum
    start_points = {
        'x': [1.0, 1.2, 1.4, 1.6, 1.8],
        'y': [-1.0, -0.8, -0.6, -0.4, -0.2]
    }

    result = optimize_function(
        quadratic,
        bounds,
        start_points=start_points,
        bond_dim=4  # Must have at least bond_dim points
    )

Method Selection
----------------

The optimizer automatically chooses the best tensor network structure based on dimensionality:

.. code-block:: python

    # For low-dimensional problems (< 20 dimensions)
    # Uses tensor train network (default)
    bounds_3d = {f'x{i}': (-5, 5) for i in range(3)}
    result = optimize_function(my_func, bounds_3d, method='auto')

    # For high-dimensional problems (>= 20 dimensions)
    # Automatically uses balanced tree network
    bounds_25d = {f'x{i}': (-5, 5) for i in range(25)}
    result = optimize_function(my_func, bounds_25d, method='auto')

You can also manually specify the method:

.. code-block:: python

    # Force tensor train
    result = optimize_function(my_func, bounds, method='tensor_train')

    # Force balanced tree
    result = optimize_function(my_func, bounds, method='balanced_tree')

Hyperparameter Tuning Example
------------------------------

A practical example for machine learning hyperparameter optimization:

.. code-block:: python

    from qutree import optimize_function

    # Define your objective (e.g., validation error)
    def model_error(learning_rate, batch_size, dropout_rate, l2_reg):
        # Replace with your actual model training/validation
        # This is a placeholder that returns simulated error
        error = (learning_rate - 0.001)**2 + (batch_size - 64)**2 / 1000
        error += (dropout_rate - 0.3)**2 + (l2_reg - 0.01)**2
        return error

    # Define hyperparameter search space
    bounds = {
        'learning_rate': (1e-5, 1e-1),
        'batch_size': (16, 128),
        'dropout_rate': (0.0, 0.5),
        'l2_reg': (1e-6, 1e-1)
    }

    # Use different grid resolutions for different parameters
    grid_points = {
        'learning_rate': 25,  # Fine grid
        'batch_size': 15,     # Coarse grid
        'dropout_rate': 11,   # Coarse grid
        'l2_reg': 21          # Medium grid
    }

    # Optimize hyperparameters
    result = optimize_function(
        model_error,
        bounds,
        grid_points=grid_points,
        n_sweeps=5
    )

    print("Best hyperparameters:")
    for param, value in result['x'].items():
        print(f"  {param}: {value:.6f}")
    print(f"Best validation error: {result['fun']:.6f}")
    print(f"Function evaluations: {result['n_calls']}")

scipy.optimize-like Interface
------------------------------

For users familiar with scipy, the ``minimize`` function provides a similar interface:

.. code-block:: python

    from qutree import minimize

    def sphere(x):
        return np.sum(x**2)

    # List of bounds like scipy.optimize.minimize
    bounds = [(-5, 5), (-5, 5), (-5, 5)]

    result = minimize(sphere, bounds, n_sweeps=3, bond_dim=4)

    # Access results
    print(f"Optimal value: {result['fun']}")
    print(f"Optimal point: {result['x']}")  # Dict with x0, x1, x2

Parameters Reference
--------------------

**optimize_function(func, bounds, ...)**

Parameters:
    * **func** : callable
        Function to minimize. Can accept named parameters, dict, or array.

    * **bounds** : dict
        Dictionary mapping parameter names to (min, max) tuples.
        Example: ``{'x': (0, 10), 'y': (-5, 5)}``

    * **method** : str, default='auto'
        Optimization method: 'auto', 'tensor_train' ('tt'), or 'balanced_tree' ('bt')

    * **n_sweeps** : int, default=3
        Number of optimization sweeps. More sweeps = better convergence.

    * **bond_dim** : int, default=4
        Bond dimension r. Larger values = more accurate but slower.

    * **grid_points** : int or dict, default=21
        Number of grid points per dimension. Can be:

        - int: Same number for all dimensions
        - dict: Per-parameter grid points, e.g. ``{'x': 21, 'y': 31}``

    * **start_points** : dict, optional
        Initial guess points for warm-start.
        Format: ``{'x': [x1, x2, ...], 'y': [y1, y2, ...]}``
        Each parameter needs >= bond_dim points.

    * **verbose** : bool, default=True
        Print optimization progress and results.

Returns:
    Dictionary containing:
        * **x** : dict of optimal parameter values
        * **fun** : optimal function value
        * **n_calls** : number of function evaluations
        * **n_cache_hits** : number of cache hits
        * **objective** : Objective instance with full history
        * **network** : optimized tensor network

Tips and Best Practices
------------------------

**Grid Resolution**

- Start with 11-21 grid points for initial exploration
- Increase to 31-51 for final refinement
- Use per-parameter grids for efficiency in high dimensions

**Number of Sweeps**

- 2-3 sweeps often sufficient for simple functions
- 5-10 sweeps for complex landscapes
- Monitor convergence by checking if result improves

**Bond Dimension**

- Start with bond_dim=4 (default)
- Increase to 6-8 for more accurate results
- Higher bond dimensions increase computational cost

**High-Dimensional Problems**

- Use balanced tree for >= 20 dimensions
- Use coarser grids (11 points) to keep cost manageable
- Consider per-parameter grids if some dimensions are more important

**Debugging**

- Set ``verbose=True`` to see optimization progress
- Check ``result['n_calls']`` to understand computational cost
- Examine ``result['objective'].logger.df`` for full optimization history

See Also
--------

* :doc:`ttnopt` - Low-level TTNOpt interface
* :doc:`tree_structures` - Understanding tensor network structures
* :doc:`../examples` - More usage examples
