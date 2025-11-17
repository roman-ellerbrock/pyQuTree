pyQuTree Documentation
======================

**pyQuTree** is a tree tensor network package for optimization, quantum chemistry, and scientific computing.

Quick Example
-------------

.. code-block:: python

    from qutree import optimize_function

    def rosenbrock(x, y):
        return (1 - x)**2 + 100*(y - x**2)**2

    bounds = {'x': (-2, 2), 'y': (-1, 3)}
    result = optimize_function(rosenbrock, bounds)

Features
--------

**High-Level Interface:**

* Easy-to-use ``optimize_function`` for arbitrary Python functions
* Support for named parameters, dicts, or arrays
* Per-parameter grid resolutions for efficiency
* Warm-start optimization with initial guesses
* Automatic method selection based on dimensionality

**Low-Level Tensor Network Methods:**

* Tensor Train (TT) networks
* Balanced tree tensor networks
* Tree tensor network optimization (TTNOpt)
* Cross approximation methods
* Simultaneous diagonalization

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   usage/convenience_interface
   usage/ttnopt
   usage/tree_structures
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
