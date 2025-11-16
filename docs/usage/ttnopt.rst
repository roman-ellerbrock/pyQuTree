Tree Tensor Network Optimization (TTNOpt)
==========================================

The ``ttnopt`` function is the core optimization routine in pyQuTree. It uses tree tensor network structures to efficiently explore high-dimensional parameter spaces.

Overview
--------

TTNOpt performs alternating least-squares optimization on a tree tensor network to find the minimum of an objective function. The method exploits the hierarchical structure of tree tensor networks to achieve better scaling than traditional grid-based methods.

Basic Usage
-----------

The basic signature of ``ttnopt`` is:

.. code-block:: python

    from qutree import ttnopt, Objective

    G_opt = ttnopt(G, objective, nsweeps, primitive_grid, start_grid=None)

Parameters
~~~~~~~~~~

* **G**: The tree tensor network graph structure (created with ``tensor_train_graph`` or ``balanced_tree``)
* **objective**: An ``Objective`` wrapper around your function to minimize
* **nsweeps**: Number of optimization sweeps through the tree
* **primitive_grid**: Grid boundaries for each dimension
* **start_grid** (optional): Initial points for optimization (shape: f x r or f x m where m >= r)

Complete Example
----------------

Here's a complete example optimizing a simple quadratic function:

.. code-block:: python

    from qutree import *
    import numpy as np

    # Define the function to minimize
    def V(x):
        """Sum of squared distances from target points [0, 1, 2]"""
        point = np.array([0.0, 1.0, 2.0])
        return np.sum((x - point)**2)

    # Wrap in Objective
    objective = Objective(V)

    # Define parameters
    N = 21       # Grid points per dimension
    r = 4        # Bond dimension (controls accuracy)
    f = 3        # Number of dimensions
    nsweeps = 3  # Number of optimization sweeps

    # Create tensor train network
    G = tensor_train_graph(f, r, N)

    # Define grid boundaries [min, max] with N points
    primitive_grid = [np.linspace(0., 4., num=N)] * f

    # Run optimization
    G_opt = ttnopt(G, objective, nsweeps, primitive_grid)

    # Access results
    print(f"Number of function calls: {objective.ncalls}")
    print(f"Minimum found: {objective.logger.df.loc[objective.logger.df['f'].idxmin()]}")

Using Start Points
------------------

You can provide initial guess points to warm-start the optimization:

.. code-block:: python

    # Provide starting points (3 dimensions x 4+ points)
    start_grid = np.array([
        [0., 0., 0., 0.5],  # x1 initial points
        [1., 2., 4., 1.5],  # x2 initial points
        [2., 5., 3., 2.5],  # x3 initial points
    ])

    G_opt = ttnopt(G, objective, nsweeps, primitive_grid, start_grid)

The start grid should have shape ``(f, m)`` where:

* ``f`` is the number of dimensions
* ``m >= r`` (at least as many points as the bond dimension)

Good starting points can significantly reduce the number of sweeps needed for convergence.

Tracking Progress
-----------------

The ``Objective`` wrapper includes a logger that tracks all function evaluations:

.. code-block:: python

    import pandas as pd

    # After optimization
    df = objective.logger.df

    # View optimization progress
    print(df.head())

    # Find best point
    best_idx = df['f'].idxmin()
    best_point = df.loc[best_idx, ['x1', 'x2', 'x3']]
    best_value = df.loc[best_idx, 'f']

    print(f"Best point: {best_point.values}")
    print(f"Best value: {best_value}")

    # Plot convergence
    import matplotlib.pyplot as plt
    plt.plot(df['f'])
    plt.xlabel('Evaluation')
    plt.ylabel('Objective Value')
    plt.yscale('log')
    plt.show()

Understanding Sweeps
--------------------

A "sweep" is a complete pass through all nodes in the tensor network. During each sweep:

1. The algorithm visits each node in the tree
2. At each node, it optimizes the local tensor while keeping others fixed
3. Results are cached to avoid redundant function evaluations

Typical behavior:

* **Sweep 0**: Initial exploration of the space
* **Sweep 1**: Refinement around promising regions
* **Sweep 2+**: Fine-tuning and convergence

Most problems converge within 2-5 sweeps. Monitor the objective value to determine if more sweeps are needed.

Performance Considerations
---------------------------

Bond Dimension (r)
~~~~~~~~~~~~~~~~~~

The bond dimension controls the expressiveness of the tensor network:

* **Small r (2-4)**: Fast but may miss complex features
* **Medium r (4-8)**: Good balance for most problems
* **Large r (>8)**: Higher accuracy but slower

Grid Resolution (N)
~~~~~~~~~~~~~~~~~~~

The number of grid points per dimension:

* **Small N (<20)**: Fast coarse optimization
* **Medium N (20-50)**: Standard resolution
* **Large N (>50)**: High precision, slower

Rule of thumb: Start with small r and N, increase as needed.

Dimensionality (f)
~~~~~~~~~~~~~~~~~~

TTNOpt scales well with dimensionality:

* Traditional grid: O(N^f) - exponential
* Tensor train: O(f × N × r²) - linear in f
* Balanced tree: O(f × N × r + r³ × log₂(f)) - logarithmic in f

For high-dimensional problems (f > 10), use balanced trees for better scaling.

Caching
~~~~~~~

The ``Objective`` wrapper automatically caches function evaluations:

.. code-block:: python

    # View cache statistics
    print(f"Direct calls: {objective.ncalls}")
    print(f"Cache hits: {objective.ncache}")
    print(f"Total accesses: {objective.ncalls + objective.ncache}")

Efficient caching is crucial for expensive objective functions.

Common Issues
-------------

Optimization Not Converging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the optimization doesn't find a good minimum:

1. Increase ``nsweeps`` (try 5-10)
2. Increase bond dimension ``r``
3. Check that ``primitive_grid`` covers the region containing the minimum
4. Provide good ``start_grid`` points if you have prior knowledge

High Function Call Count
~~~~~~~~~~~~~~~~~~~~~~~~~

If too many function evaluations are needed:

1. Reduce ``r`` (bond dimension)
2. Reduce ``N`` (grid resolution)
3. Use fewer ``nsweeps``
4. Ensure caching is working properly

Memory Issues
~~~~~~~~~~~~~

For very large problems:

1. Use balanced trees instead of tensor trains
2. Reduce ``r`` and ``N``
3. Process data in batches if possible

See Also
--------

* :doc:`tree_structures` - Learn about different network topologies
* :doc:`../api/index` - Full API reference
