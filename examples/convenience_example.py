"""
Example demonstrating the high-level convenience interface for function optimization.

This shows how to optimize arbitrary Python functions using the optimize_function
and minimize interfaces without needing to know the details of tensor networks.
"""
import numpy as np
from qutree import optimize_function, minimize


def example_1_rosenbrock():
    """
    Example 1: Optimize the Rosenbrock function.

    The Rosenbrock function has a global minimum at (1, 1) with value 0.
    """
    print("=" * 70)
    print("Example 1: Rosenbrock Function")
    print("=" * 70)

    def rosenbrock(x, y):
        """The famous Rosenbrock banana function."""
        return (1 - x)**2 + 100*(y - x**2)**2

    bounds = {'x': (-2, 2), 'y': (-1, 3)}

    result = optimize_function(rosenbrock, bounds, n_sweeps=3, bond_dim=6)

    print(f"\nResult: x={result['x']['x']:.3f}, y={result['x']['y']:.3f}")
    print(f"(True minimum is at x=1, y=1)")
    print()


def example_2_multidimensional():
    """
    Example 2: Optimize a higher-dimensional sphere function.

    Demonstrates optimization in 10 dimensions.
    """
    print("=" * 70)
    print("Example 2: 10-Dimensional Sphere Function")
    print("=" * 70)

    def sphere(**kwargs):
        """Sphere function with minimum at all zeros."""
        return sum(v**2 for v in kwargs.values())

    # Create bounds for 10 dimensions
    bounds = {f'x{i}': (-5, 5) for i in range(10)}

    result = optimize_function(
        sphere,
        bounds,
        method='tensor_train',  # Explicitly use tensor train for <20 dimensions
        n_sweeps=3,
        bond_dim=4,
        grid_points=21,
        verbose=True
    )

    print(f"\nOptimal value: {result['fun']:.6f}")
    print(f"(True minimum is 0 at all x_i = 0)")
    print()


def example_3_warm_start():
    """
    Example 3: Warm-start optimization with initial guess.

    Demonstrates how to provide initial points to speed up convergence.
    """
    print("=" * 70)
    print("Example 3: Quadratic with Warm Start")
    print("=" * 70)

    def quadratic(a, b, c):
        """Quadratic with minimum at (2, -1, 3)."""
        return (a - 2)**2 + (b + 1)**2 + (c - 3)**2

    bounds = {'a': (0, 4), 'b': (-3, 1), 'c': (1, 5)}

    # Provide initial guess near the optimum
    start_points = {
        'a': [1.5, 1.8, 2.0, 2.2, 2.5],
        'b': [-1.5, -1.2, -1.0, -0.8, -0.5],
        'c': [2.5, 2.7, 3.0, 3.3, 3.5]
    }

    result = optimize_function(
        quadratic,
        bounds,
        start_points=start_points,
        n_sweeps=2,
        bond_dim=4,
        verbose=True
    )

    print(f"\nResult: a={result['x']['a']:.3f}, b={result['x']['b']:.3f}, c={result['x']['c']:.3f}")
    print(f"(True minimum is at a=2, b=-1, c=3)")
    print()


def example_4_minimize_interface():
    """
    Example 4: Using the scipy-like minimize interface.

    Demonstrates the minimize() function with list-based bounds.
    """
    print("=" * 70)
    print("Example 4: scipy-like minimize() Interface")
    print("=" * 70)

    def objective(x):
        """Objective function taking array input."""
        return np.sum((x - np.array([1, 2, 3]))**2)

    # Can use list of bounds like scipy.optimize.minimize
    bounds = [(-1, 3), (0, 4), (1, 5)]

    result = minimize(
        objective,
        bounds,
        n_sweeps=2,
        bond_dim=4,
        verbose=True
    )

    print(f"\nOptimal value: {result['fun']:.6f}")
    print(f"Optimal point: x0={result['x']['x0']:.3f}, x1={result['x']['x1']:.3f}, x2={result['x']['x2']:.3f}")
    print(f"(True minimum is at [1, 2, 3])")
    print()


def example_5_automatic_method():
    """
    Example 5: Automatic method selection.

    The 'auto' method chooses tensor train for <20 dimensions,
    balanced tree for >=20 dimensions.
    """
    print("=" * 70)
    print("Example 5: Automatic Method Selection")
    print("=" * 70)

    def high_dim_sphere(**kwargs):
        """Sphere function for any dimension."""
        return sum(v**2 for v in kwargs.values())

    # 25-dimensional problem - will automatically use balanced tree
    bounds = {f'x{i}': (-2, 2) for i in range(25)}

    result = optimize_function(
        high_dim_sphere,
        bounds,
        method='auto',  # Will choose balanced_tree since dimension >= 20
        n_sweeps=2,
        bond_dim=4,
        grid_points=11,  # Use coarser grid for high dimensions
        verbose=True
    )

    print(f"\nOptimal value: {result['fun']:.6f}")
    print(f"Number of function evaluations: {result['n_calls']}")
    print()


def example_6_per_parameter_grid():
    """
    Example 6: Using different grid resolutions for different parameters.

    Demonstrates per-parameter grid points for more efficient optimization.
    """
    print("=" * 70)
    print("Example 6: Per-Parameter Grid Points")
    print("=" * 70)

    def anisotropic(x, y, z):
        """Function with different sensitivities in different directions."""
        # y has a narrow valley, needs finer resolution
        return (x - 1)**2 + 100*(y - 0.5)**2 + (z + 1)**2

    bounds = {'x': (-2, 3), 'y': (-1, 2), 'z': (-3, 1)}

    # Use different grid resolutions for different parameters
    # Fine grid (31 points) for y (narrow valley)
    # Coarse grid (11 points) for x and z (wider valleys)
    grid_points = {
        'x': 11,  # Coarse grid
        'y': 31,  # Fine grid for narrow valley
        'z': 11   # Coarse grid
    }

    result = optimize_function(
        anisotropic,
        bounds,
        grid_points=grid_points,
        n_sweeps=3,
        bond_dim=4,
        verbose=True
    )

    print(f"\nResult: x={result['x']['x']:.3f}, y={result['x']['y']:.3f}, z={result['x']['z']:.3f}")
    print(f"(True minimum is at x=1, y=0.5, z=-1)")
    print(f"Using per-parameter grids saved {(31*31*31) - (11*31*11):,} function evaluations")
    print(f"vs. uniform 31x31x31 grid")
    print()


if __name__ == '__main__':
    # Run all examples
    example_1_rosenbrock()
    example_2_multidimensional()
    example_3_warm_start()
    example_4_minimize_interface()
    example_5_automatic_method()
    example_6_per_parameter_grid()

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
