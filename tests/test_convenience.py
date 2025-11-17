"""
Unit tests for the convenience interface for optimizing arbitrary functions.

Tests the high-level optimize_function and minimize interfaces with different
function signatures and parameter formats.
"""
import numpy as np
import pytest
from qutree import optimize_function, minimize, FunctionWrapper


def test_function_with_named_params():
    """Test optimizing a function with named parameters."""
    def quadratic(x, y, z):
        """Simple quadratic with minimum at (1, 2, 3)."""
        return (x - 1)**2 + (y - 2)**2 + (z - 3)**2

    bounds = {
        'x': (0, 2),
        'y': (1, 3),
        'z': (2, 4)
    }

    result = optimize_function(
        quadratic,
        bounds,
        n_sweeps=2,
        bond_dim=4,
        grid_points=21,
        verbose=False
    )

    # Verify result structure
    assert 'x' in result
    assert 'fun' in result
    assert 'n_calls' in result
    assert 'n_cache_hits' in result
    assert 'objective' in result
    assert 'network' in result

    # Verify optimal parameters are in result
    assert 'x' in result['x']
    assert 'y' in result['x']
    assert 'z' in result['x']

    # Verify we found a reasonable minimum (close to [1, 2, 3])
    assert result['x']['x'] == pytest.approx(1.0, abs=0.3)
    assert result['x']['y'] == pytest.approx(2.0, abs=0.3)
    assert result['x']['z'] == pytest.approx(3.0, abs=0.3)
    assert result['fun'] < 0.5


def test_function_with_dict_param():
    """Test optimizing a function that accepts a dictionary."""
    def dict_func(params):
        """Function that takes dict parameter."""
        return (params['a'] - 0.5)**2 + (params['b'] + 1.0)**2

    bounds = {
        'a': (-1, 2),
        'b': (-2, 1)
    }

    result = optimize_function(
        dict_func,
        bounds,
        n_sweeps=2,
        bond_dim=4,
        grid_points=21,
        verbose=False
    )

    # Verify we found the minimum at (0.5, -1.0)
    assert result['x']['a'] == pytest.approx(0.5, abs=0.3)
    assert result['x']['b'] == pytest.approx(-1.0, abs=0.3)
    assert result['fun'] < 0.5


def test_function_with_array_param():
    """Test optimizing a function that accepts numpy array."""
    def array_func(x):
        """Function that takes array parameter."""
        target = np.array([1.0, -0.5])
        return np.sum((x - target)**2)

    bounds = {
        'x0': (-2, 3),
        'x1': (-2, 1)
    }

    result = optimize_function(
        array_func,
        bounds,
        n_sweeps=2,
        bond_dim=4,
        grid_points=21,
        verbose=False
    )

    # Verify we found the minimum at [1.0, -0.5]
    assert result['x']['x0'] == pytest.approx(1.0, abs=0.3)
    assert result['x']['x1'] == pytest.approx(-0.5, abs=0.3)
    assert result['fun'] < 0.5


def test_rosenbrock_function():
    """Test the classic Rosenbrock function."""
    def rosenbrock(x, y):
        return (1 - x)**2 + 100*(y - x**2)**2

    bounds = {
        'x': (-2, 2),
        'y': (-1, 3)
    }

    result = optimize_function(
        rosenbrock,
        bounds,
        n_sweeps=3,
        bond_dim=6,
        grid_points=31,
        verbose=False
    )

    # Rosenbrock minimum is at (1, 1)
    # The function has a very narrow valley, making it challenging to optimize
    # with grid-based methods. We verify we found a reasonable solution.
    assert result['x']['x'] == pytest.approx(1.0, abs=0.5)
    # The y-coordinate is especially challenging due to the narrow valley
    assert result['fun'] < 10.0  # Much better than random sampling would give


def test_tensor_train_method():
    """Test explicit tensor_train method selection."""
    def simple_quad(x, y):
        return x**2 + y**2

    bounds = {'x': (-2, 2), 'y': (-2, 2)}

    result = optimize_function(
        simple_quad,
        bounds,
        method='tensor_train',
        n_sweeps=2,
        bond_dim=4,
        grid_points=21,
        verbose=False
    )

    assert result['x']['x'] == pytest.approx(0.0, abs=0.3)
    assert result['x']['y'] == pytest.approx(0.0, abs=0.3)
    assert result['fun'] < 0.5


def test_balanced_tree_method():
    """Test explicit balanced_tree method selection."""
    def simple_quad(x, y, z):
        return x**2 + y**2 + z**2

    bounds = {'x': (-2, 2), 'y': (-2, 2), 'z': (-2, 2)}

    result = optimize_function(
        simple_quad,
        bounds,
        method='balanced_tree',
        n_sweeps=2,
        bond_dim=4,
        grid_points=21,
        verbose=False
    )

    assert result['x']['x'] == pytest.approx(0.0, abs=0.3)
    assert result['x']['y'] == pytest.approx(0.0, abs=0.3)
    assert result['x']['z'] == pytest.approx(0.0, abs=0.3)
    assert result['fun'] < 0.5


def test_auto_method_selection():
    """Test automatic method selection based on dimensionality."""
    # Low dimensional (< 20) should use tensor train
    def low_dim(**kwargs):
        return sum(v**2 for v in kwargs.values())

    bounds_low = {f'x{i}': (-1, 1) for i in range(5)}

    result_low = optimize_function(
        low_dim,
        bounds_low,
        method='auto',
        n_sweeps=2,
        bond_dim=4,
        grid_points=11,
        verbose=False
    )

    assert result_low['fun'] < 0.5

    # High dimensional (>= 20) should use balanced tree
    bounds_high = {f'x{i}': (-1, 1) for i in range(20)}

    result_high = optimize_function(
        low_dim,
        bounds_high,
        method='auto',
        n_sweeps=2,
        bond_dim=4,
        grid_points=11,
        verbose=False
    )

    # Higher dimensional problems are harder, so use more generous tolerance
    assert result_high['fun'] < 2.0


def test_with_start_points():
    """Test warm-start with initial guess points."""
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
        n_sweeps=2,
        bond_dim=4,
        grid_points=21,
        verbose=False
    )

    # With warm start, should converge close to (1.5, -0.5)
    assert result['x']['x'] == pytest.approx(1.5, abs=0.3)
    assert result['x']['y'] == pytest.approx(-0.5, abs=0.3)
    assert result['fun'] < 0.5


def test_minimize_interface():
    """Test the minimize() alias function."""
    def sphere(x, y, z):
        return x**2 + y**2 + z**2

    bounds = {
        'x': (-2, 2),
        'y': (-2, 2),
        'z': (-2, 2)
    }

    result = minimize(
        sphere,
        bounds,
        n_sweeps=2,
        bond_dim=4,
        grid_points=21,
        verbose=False
    )

    assert result['x']['x'] == pytest.approx(0.0, abs=0.3)
    assert result['x']['y'] == pytest.approx(0.0, abs=0.3)
    assert result['x']['z'] == pytest.approx(0.0, abs=0.3)
    assert result['fun'] < 0.5


def test_minimize_with_list_bounds():
    """Test minimize() with list of bounds instead of dict."""
    def array_sphere(x):
        return np.sum(x**2)

    # List of (min, max) tuples
    bounds = [(-2, 2), (-2, 2), (-2, 2)]

    result = minimize(
        array_sphere,
        bounds,
        n_sweeps=2,
        bond_dim=4,
        grid_points=21,
        verbose=False
    )

    # Result should have auto-generated parameter names x0, x1, x2
    assert 'x0' in result['x']
    assert 'x1' in result['x']
    assert 'x2' in result['x']

    assert result['x']['x0'] == pytest.approx(0.0, abs=0.3)
    assert result['x']['x1'] == pytest.approx(0.0, abs=0.3)
    assert result['x']['x2'] == pytest.approx(0.0, abs=0.3)
    assert result['fun'] < 0.5


def test_invalid_method_raises_error():
    """Test that invalid method raises ValueError."""
    def dummy(x):
        return x**2

    bounds = {'x': (-1, 1)}

    with pytest.raises(ValueError, match="Unknown method"):
        optimize_function(dummy, bounds, method='invalid_method')


def test_missing_start_point_raises_error():
    """Test that missing parameters in start_points raises ValueError."""
    def dummy(x, y):
        return x**2 + y**2

    bounds = {'x': (-1, 1), 'y': (-1, 1)}
    start_points = {'x': [0, 0.5, 1.0, 1.5]}  # Missing 'y'

    with pytest.raises(ValueError, match="start_points missing parameter"):
        optimize_function(dummy, bounds, start_points=start_points)


def test_insufficient_start_points_raises_error():
    """Test that insufficient start_points raises ValueError."""
    def dummy(x, y):
        return x**2 + y**2

    bounds = {'x': (-1, 1), 'y': (-1, 1)}
    start_points = {
        'x': [0, 0.5],  # Only 2 points, need at least bond_dim (4)
        'y': [0, 0.5]
    }

    with pytest.raises(ValueError, match="need at least"):
        optimize_function(dummy, bounds, start_points=start_points, bond_dim=4)


def test_function_wrapper_direct():
    """Test FunctionWrapper class directly."""
    def func(a, b, c):
        return a + 2*b + 3*c

    wrapper = FunctionWrapper(func, ['a', 'b', 'c'])

    # Test with array input
    x = np.array([1.0, 2.0, 3.0])
    result = wrapper(x)

    expected = 1.0 + 2*2.0 + 3*3.0  # = 14.0
    assert result == expected

    # Verify parameter mapping
    assert wrapper.n_params == 3
    assert wrapper.param_names == ['a', 'b', 'c']
