"""
Unit tests for TTNOpt examples from documentation.

Tests the examples from README and documentation to ensure they run correctly
with both tensor train and balanced tree network structures.
"""
import numpy as np
from qutree import *


def test_readme_example():
    """Test the example from README.md - closely matches the actual README code."""
    def V(x):
        # change with your objective function
        return np.sum((x - np.ones(x.shape[0]))**2)

    N, r, f, nsweep = 21, 4, 3, 2  # Reduced nsweep for faster tests

    objective = Objective(V)

    # Create a tensor network using tensor train (as shown in ttopt_example.ipynb)
    tn = tensor_train_graph(f, r, N)

    # Create a primitive grid
    primitive_grid = [np.linspace(-1., 3., N)] * f

    # Tensor network optimization
    tn_updated = ttnopt(tn, objective, nsweep, primitive_grid)

    # Verify optimization ran
    assert objective.function_calls > 0, "Objective function should have been called"
    assert objective.logger.df is not None, "Logger should have recorded data"
    assert len(objective.logger.df) > 0, "Logger should have entries"

    # Check that we found a reasonable minimum (should be close to [1, 1, 1])
    best_idx = objective.logger.df['f'].idxmin()
    best_value = objective.logger.df.loc[best_idx, 'f']

    # The minimum should be close to 0 (at x = [1, 1, 1])
    assert best_value < 1.0, f"Expected minimum < 1.0, got {best_value}"


def test_tensor_train_example():
    """Test TTNOpt with tensor train network structure from ttopt_example.ipynb."""
    def V(x):
        # Minimize sum of squared deviations from target [0, 1, 2]
        point = np.array(list(range(x.shape[0])))
        return np.sum((x - point)**2)

    objective = Objective(V)

    # Parameters from ttopt_example.ipynb
    N = 21      # Grid points per dimension
    r = 4       # Bond dimension
    f = 3       # Number of dimensions
    nsweeps = 2 # Number of optimization sweeps (reduced for faster tests)

    # Create tensor train graph
    G = tensor_train_graph(f, r, N)

    # Define primitive grid boundaries
    primitive_grid = [np.linspace(0., 4., num=N)] * f

    # Run optimization
    G_opt = ttnopt(G, objective, nsweeps, primitive_grid)

    # Verify optimization ran successfully
    assert objective.function_calls > 0, "Objective function should have been called"
    assert objective.logger.df is not None, "Logger should have recorded data"
    assert len(objective.logger.df) > 0, "Logger should have entries"

    # Check that we found a good minimum
    best_idx = objective.logger.df['f'].idxmin()
    best_value = objective.logger.df.loc[best_idx, 'f']
    best_point = objective.logger.df.loc[best_idx, ['x1', 'x2', 'x3']].values

    # The minimum should be close to 0 (at x = [0, 1, 2])
    assert best_value < 1.0, f"Expected minimum < 1.0, got {best_value}"

    # Check that the solution is close to the expected minimum
    expected = np.array([0., 1., 2.])
    distance = np.linalg.norm(best_point - expected)
    assert distance < 0.5, f"Expected solution close to [0, 1, 2], got {best_point}"


def test_balanced_tree_example():
    """Test TTNOpt with balanced tree network structure."""
    def V(x):
        # change with your objective function
        return np.sum((x - np.ones(x.shape[0]))**2)

    N, r, f, nsweep = 21, 4, 3, 2  # Reduced nsweep for faster tests

    objective = Objective(V)

    # Create a balanced tree tensor network
    tn = balanced_tree(f, r, N)

    # Create a primitive grid
    primitive_grid = [np.linspace(-1., 3., N)] * f

    # Tensor network optimization
    tn_updated = ttnopt(tn, objective, nsweep, primitive_grid)

    # Verify optimization ran
    assert objective.function_calls > 0, "Objective function should have been called"
    assert objective.logger.df is not None, "Logger should have recorded data"
    assert len(objective.logger.df) > 0, "Logger should have entries"

    # Check that we found a reasonable minimum (should be close to [1, 1, 1])
    best_idx = objective.logger.df['f'].idxmin()
    best_value = objective.logger.df.loc[best_idx, 'f']

    # The minimum should be close to 0 (at x = [1, 1, 1])
    assert best_value < 1.0, f"Expected minimum < 1.0, got {best_value}"
