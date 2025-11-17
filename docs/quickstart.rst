Quick Start
===========

High-Level Interface (Recommended)
-----------------------------------

The easiest way to get started is with the convenience interface:

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
    print(f"Function evaluations: {result['n_calls']}")

That's it! The convenience interface handles all the tensor network details for you.

See :doc:`usage/convenience_interface` for more examples and advanced features like:

* Per-parameter grid points
* Warm-start optimization
* Different function signatures
* Hyperparameter tuning

Low-Level Interface
-------------------

For more control over the tensor network structure, use the low-level API:

.. code-block:: python

    from qutree import *
    import numpy as np

    # Define objective function
    def V(x):
        point = np.array(list(range(x.shape[0])))
        return np.sum((x - point)**2)

    # Create objective wrapper
    objective = Objective(V)

    # Parameters
    N = 21      # Grid points per dimension
    r = 4       # Bond dimension
    f = 3       # Number of features/dimensions
    nsweeps = 3 # Number of optimization sweeps

    # Create tensor train graph
    G = tensor_train_graph(f, r, N)

    # Define primitive grid boundaries
    primitive_grid = [np.linspace(0., 4., num=N)] * f

    # Run optimization
    G_opt = ttnopt(G, objective, nsweeps, primitive_grid)

    # Access results
    print(objective)
    print(objective.logger.df)

The optimization will find the minimum of the function V(x), which is at x = [0, 1, 2].

Visualizing the Network
-----------------------

You can visualize the tensor network structure:

.. code-block:: python

    from qutree import plot_tt_diagram, plot_tree

    # For tensor train
    fig = plot_tt_diagram(G)

    # For tree structures
    fig = plot_tree(G)

Next Steps
----------

* Start with the :doc:`Convenience Interface <usage/convenience_interface>` for easy optimization
* Learn about different :doc:`tree structures <usage/tree_structures>`
* Explore :doc:`TTNOpt optimization <usage/ttnopt>` in detail for low-level control
* Check the :doc:`API reference <api/index>`
