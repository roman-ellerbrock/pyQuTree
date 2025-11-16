# pyQuTree

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

You can use a tree tensor network version of cross interpolation[^2] via
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