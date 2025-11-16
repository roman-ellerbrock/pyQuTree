Quick Start
===========

Basic Optimization Example
---------------------------

Here's a minimal example of optimizing a 3D function using tensor train networks:

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

* Learn about different :doc:`tree structures <usage/tree_structures>`
* Explore :doc:`TTNOpt optimization <usage/ttnopt>` in detail
* Check the :doc:`API reference <api/index>`
