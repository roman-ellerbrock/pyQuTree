Tree Tensor Network Structures
===============================

pyQuTree supports different tree tensor network topologies, each with distinct properties and performance characteristics. This guide explains when and how to use each structure.

Overview
--------

Tree tensor networks decompose high-dimensional data into a hierarchical structure. The choice of topology affects:

* Computational complexity
* Memory requirements
* Convergence speed
* Scalability with dimension

Available Structures
--------------------

pyQuTree provides two main network structures:

1. **Tensor Train (TT)**: Linear chain topology
2. **Balanced Tree (BT)**: Hierarchical binary tree topology

Tensor Train Networks
---------------------

The tensor train is a linear chain where each dimension is connected sequentially.

Creating a Tensor Train
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from qutree import tensor_train_graph, plot_tt_diagram
    import numpy as np

    f = 5   # number of dimensions
    r = 4   # bond dimension
    N = 21  # grid points per dimension

    # Create tensor train graph
    G = tensor_train_graph(f, r, N)

    # Visualize the structure
    fig = plot_tt_diagram(G)

Structure
~~~~~~~~~

In a tensor train with f dimensions:

* Each dimension is represented by a node in a linear chain
* Adjacent nodes are connected with bond dimension r
* Total number of parameters: O(f × N × r²)

Visual representation for f=3::

    [x1]---r---[x2]---r---[x3]
     |          |          |
     N          N          N

Where:
* ``[xi]`` represents dimension i
* ``r`` is the bond dimension connecting adjacent nodes
* ``N`` is the number of grid points per dimension

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

* **Storage**: O(f × N × r²)
* **Optimization sweep**: O(f × N × r² × eval)
* **Scaling with f**: Linear

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

    from qutree import *
    import numpy as np

    def objective_function(x):
        # Minimize sum of squared deviations
        target = np.arange(len(x))
        return np.sum((x - target)**2)

    objective = Objective(objective_function)

    # Setup
    f, r, N, nsweeps = 10, 4, 21, 3

    # Create tensor train
    G = tensor_train_graph(f, r, N)

    # Define grid
    primitive_grid = [np.linspace(0., float(f-1), num=N)] * f

    # Optimize
    G_opt = ttnopt(G, objective, nsweeps, primitive_grid)

When to Use Tensor Trains
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Advantages:**

* Simple structure, easy to understand
* Efficient for low-to-medium dimensions (f < 20)
* Well-studied with strong theoretical guarantees
* Good for problems with sequential structure

**Disadvantages:**

* Linear scaling in f (can be slow for f > 50)
* Information must pass through the entire chain
* Asymmetric: end dimensions are treated differently than middle ones

Balanced Tree Networks
----------------------

The balanced tree organizes dimensions in a hierarchical binary tree structure.

Creating a Balanced Tree
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from qutree import balanced_tree, plot_tree
    import numpy as np

    f = 8   # number of dimensions (works best with powers of 2)
    r = 4   # bond dimension
    N = 21  # grid points per dimension

    # Create balanced tree
    G = balanced_tree(f, r, N)

    # Visualize the structure
    fig = plot_tree(G)

Structure
~~~~~~~~~

In a balanced tree with f dimensions:

* Leaf nodes represent physical dimensions
* Internal nodes combine information hierarchically
* Tree depth: O(log₂(f))
* More symmetric than tensor train

Visual representation for f=4::

           [root]
          /      \\
      [n1]        [n2]
      /  \\        /  \\
    [x1][x2]    [x3][x4]
     |    |      |    |
     N    N      N    N

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

* **Storage**: O(f × N × r + r³ × log₂(f))
* **Optimization sweep**: O(f × N × r + r³ × log₂(f) × eval)
* **Scaling with f**: Logarithmic

This is significantly better than tensor train for large f!

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

    from qutree import *
    import numpy as np

    def high_dim_function(x):
        # Example: Rosenbrock-like function
        result = 0.0
        for i in range(len(x)-1):
            result += 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return result

    objective = Objective(high_dim_function)

    # Setup for high-dimensional problem
    f, r, N, nsweeps = 32, 6, 25, 4

    # Use balanced tree for better scaling
    G = balanced_tree(f, r, N)

    # Define grid
    primitive_grid = [np.linspace(-2., 2., num=N)] * f

    # Optional: provide starting points near expected minimum
    start_grid = np.ones((f, r))  # Start near x=1

    # Optimize
    G_opt = ttnopt(G, objective, nsweeps, primitive_grid, start_grid)

When to Use Balanced Trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Advantages:**

* Logarithmic scaling in f (excellent for f > 20)
* More symmetric treatment of all dimensions
* Better for very high-dimensional problems
* Faster convergence for some problem types

**Disadvantages:**

* More complex structure
* Slightly more overhead for small f
* Works best when f is a power of 2

Comparison
----------

Complexity Comparison
~~~~~~~~~~~~~~~~~~~~~

For same rank r and dimension N:

+-------------------+-------------------------+---------------------------+
| Structure         | Storage                 | Sweep Cost                |
+===================+=========================+===========================+
| Tensor Train      | f × N × r²              | f × N × r² × eval         |
+-------------------+-------------------------+---------------------------+
| Balanced Tree     | f × N × r + r³×log₂(f)  | (f×N×r + r³×log₂(f))×eval |
+-------------------+-------------------------+---------------------------+

For large f, balanced tree is significantly more efficient!

Example: f=64, r=4, N=20

* **Tensor Train**: ~327,000 parameters
* **Balanced Tree**: ~5,500 parameters (59× smaller!)

Performance Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

Choose **Tensor Train** when:

* f < 20 (low to medium dimensions)
* Problem has sequential/chain structure
* Simplicity is important
* You need theoretical guarantees

Choose **Balanced Tree** when:

* f > 20 (high dimensions)
* f is a power of 2 (optimal structure)
* Memory is constrained
* Problem has hierarchical structure

Example Benchmark
~~~~~~~~~~~~~~~~~

Minimizing f-dimensional quadratic function (x - target)²:

.. code-block:: python

    import time
    from qutree import *
    import numpy as np

    def benchmark(f, structure='tt'):
        def V(x):
            target = np.arange(len(x))
            return np.sum((x - target)**2)

        objective = Objective(V)
        r, N, nsweeps = 4, 21, 3

        if structure == 'tt':
            G = tensor_train_graph(f, r, N)
        else:
            G = balanced_tree(f, r, N)

        primitive_grid = [np.linspace(0., float(f), num=N)] * f

        start = time.time()
        G_opt = ttnopt(G, objective, nsweeps, primitive_grid)
        elapsed = time.time() - start

        return elapsed, objective.ncalls

    # Compare for different dimensions
    for f in [8, 16, 32, 64]:
        tt_time, tt_calls = benchmark(f, 'tt')
        bt_time, bt_calls = benchmark(f, 'bt')

        print(f"f={f:2d}: TT={tt_time:.2f}s ({tt_calls} calls), "
              f"BT={bt_time:.2f}s ({bt_calls} calls)")

Advanced: Custom Tree Structures
---------------------------------

For specialized applications, you can build custom tree structures using networkx:

.. code-block:: python

    import networkx as nx
    from qutree import TensorNetwork

    # Create custom graph
    G = nx.DiGraph()

    # Add nodes with attributes
    # Leaf nodes: have 'primitive_grid' attribute
    # Internal nodes: have 'rank' attribute

    # Example: star topology (one central node connected to all leaves)
    G.add_node(0, rank=4)  # central node

    for i in range(1, f+1):
        G.add_node(i, primitive_grid=np.linspace(0., 4., 21))
        G.add_edge(0, i)  # connect to center

    # Use in optimization
    # (Note: this is advanced usage, standard structures are recommended)

Best Practices
--------------

1. **Start Simple**: Begin with tensor train for initial experiments
2. **Scale Up**: Switch to balanced tree when f > 20
3. **Bond Dimension**: Start with r=4, increase if needed
4. **Grid Resolution**: Start with N=21, adjust based on precision needs
5. **Monitor Convergence**: Track objective value across sweeps
6. **Use Start Points**: Warm-start with domain knowledge when possible

Example Workflow
~~~~~~~~~~~~~~~~

.. code-block:: python

    from qutree import *
    import numpy as np

    # 1. Define your problem
    def my_objective(x):
        return np.sum(x**2)  # simple example

    objective = Objective(my_objective)

    # 2. Choose structure based on dimensionality
    f = 50  # high-dimensional -> use balanced tree

    # 3. Start with conservative parameters
    r, N, nsweeps = 4, 21, 3

    # 4. Create appropriate structure
    G = balanced_tree(f, r, N) if f > 20 else tensor_train_graph(f, r, N)

    # 5. Define search space
    primitive_grid = [np.linspace(-5., 5., num=N)] * f

    # 6. Run optimization
    G_opt = ttnopt(G, objective, nsweeps, primitive_grid)

    # 7. Check convergence
    df = objective.logger.df
    print(f"Best value: {df['f'].min()}")
    print(f"Function calls: {objective.ncalls}")

    # 8. If not converged, increase nsweeps or r

See Also
--------

* :doc:`ttnopt` - Detailed optimization guide
* :doc:`../quickstart` - Getting started tutorial
* :doc:`../api/index` - API reference
