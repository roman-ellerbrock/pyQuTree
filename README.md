# pyQuTree

[![Tests](https://github.com/roman-ellerbrock/pyQuTree/actions/workflows/tests.yml/badge.svg)](https://github.com/roman-ellerbrock/pyQuTree/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/pyqutree-ttn/badge/?version=latest)](https://pyqutree-ttn.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/pyqutree.svg)](https://badge.fury.io/py/pyqutree)

A smaller python version of the Tree Tensor Network library Qutree[^1]
currently centered around optimization.

## Documentation

Full documentation is available at [https://pyqutree-ttn.readthedocs.io](https://pyqutree-ttn.readthedocs.io)

## Installation

Install pyQuTree from PyPI:
```bash
pip install pyqutree
```

Or install the latest development version from GitHub:
```bash
pip install git+https://github.com/roman-ellerbrock/pyQuTree.git
```

For developers, create a conda environment via:
```bash
conda env {create, update} --file environment.yml
conda activate qutree
```

## Usage

### High-Level Interface

For quick optimization of arbitrary functions, use the convenience interface:

```python
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
```

**Practical Example - Hyperparameter Tuning:**

```python
from qutree import optimize_function
import numpy as np

# Define your objective (e.g., validation error as a function of hyperparameters)
def model_error(learning_rate, batch_size, dropout_rate, l2_reg):
    """Simulate model validation error (replace with your actual model training)."""
    # This is a placeholder - replace with your actual model training/validation
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
# (finer grid for parameters you want to optimize more precisely)
grid_points = {
    'learning_rate': 25,  # Fine grid for learning rate
    'batch_size': 15,     # Coarse grid for batch size
    'dropout_rate': 11,   # Coarse grid for dropout
    'l2_reg': 21          # Medium grid for regularization
}

# Optimize hyperparameters
result = optimize_function(model_error, bounds, grid_points=grid_points, n_sweeps=5)

print("Best hyperparameters:")
for param, value in result['x'].items():
    print(f"  {param}: {value:.6f}")
print(f"Best validation error: {result['fun']:.6f}")
print(f"Function evaluations: {result['n_calls']}")
```

### Low-Level Interface

You can also use the low-level tree tensor network API directly:

```python
from qutree import *

def V(x):
    # change with your objective function
    return np.sum((x-np.ones(x.shape[0]))**2)

N, r, f, nsweep = 21, 4, 3, 6

objective = Objective(V)

# create a tensor network, e.g. a balanced tree
tn = balanced_tree(f, r, N)

# Create a primitive grid and tensor network grid
primitive_grid = [linspace(-1., 3., N)] * f

# tensor network optimization
tn_updated = ttnopt(tn, objective, nsweep, primitive_grid)
print(objective)
dataframe = objective.logger.df
print(dataframe)
```

For detailed tutorials and usage examples, see the [documentation](https://pyqutree-ttn.readthedocs.io):
- [Quick Start Guide](https://pyqutree-ttn.readthedocs.io/en/latest/quickstart.html)
- [TTNOpt Optimization](https://pyqutree-ttn.readthedocs.io/en/latest/usage/ttnopt.html)
- [Tree Structures Guide](https://pyqutree-ttn.readthedocs.io/en/latest/usage/tree_structures.html)

More examples can be found in `examples/ttopt_example.ipynb`.

If Qutree was useful in your work, please consider citing the paper[^1].

## References
[^1] Roman Ellerbrock, K. Grace Johnson, Stefan Seritan, Hannes Hoppe, J. H. Zhang, Tim Lenzen, Thomas Weike, Uwe Manthe, Todd J. Martínez; QuTree: A tree tensor network package. J. Chem. Phys. 21 March 2024; 160 (11): 112501. https://doi.org/10.1063/5.0180233

[^2] I created the present tree tensor network version which is currently unpublished. It is inspired by Ivan Oseledets, Eugene Tyrtyshnikov, TT-cross approximation for multidimensional arrays, Linear Algebra and its Applications, Volume 432, Issue 1, 2010, Pages 70-88, https://doi.org/10.1016/j.laa.2009.07.024.