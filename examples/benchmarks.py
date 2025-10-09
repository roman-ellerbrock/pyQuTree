# benchmarking tensor cross, matrix train and ttopt optimizers

import math
import numpy as np
import pandas as pd

from qutree import (
    Grid,
    Objective,
    tensor_train_graph,
    ttnopt,
)
from qutree.optimization import (
    TensorRankOptimization,
    MatrixTrainOptimization,
    random_grid_points,
)

# test functions
def ackley(x):
    # Global min 0 at x=0, domain [-32.768, 32.768]^D
    x = np.asarray(x)
    D = x.size
    a, b, c = 20.0, 0.2, 2*np.pi
    s1 = np.sum(x**2) / D
    s2 = np.sum(np.cos(c*x)) / D
    return -a * np.exp(-b * np.sqrt(s1)) - np.exp(s2) + a + math.e

def alpine1(x):
    # Alpine #1 (multimodal), domain [-10, 10]^D, global min 0 at 0
    x = np.asarray(x)
    return np.sum(np.abs(x*np.sin(x) + 0.1*x))

def brown(x):
    # Brown function, domain [-1, 4]^D, global min 0 at 0
    x = np.asarray(x)
    s = 0.0
    for i in range(x.size-1):
        xi, xj = x[i], x[i+1]
        s += (xi**2)**(xj**2 + 1) + (xj**2)**(xi**2 + 1)
    return s

def exp_neg_norm2(x):
    # Exponential: domain [-1, 1]^D, global min -1 at 0
    x = np.asarray(x)
    return -np.exp(-np.sum(x**2))

def rosenbrock(x, a=1.0, b=100.0):
    x = np.asarray(x, dtype=float)
    return np.sum(b * (x[1:] - x[:-1]**2)**2 + (a - x[:-1])**2)

def multi_well(x, m=5, seed=42):
    """
    Linear combination of m student-t wells with negative coefficients and different heights.

    Creates local minima distributed throughout the space. Each well has a different
    amplitude (height) so they can be distinguished by a minimizer.

    Parameters:
    - x: input vector
    - m: number of wells (default 5)
    - seed: random seed for reproducible well placement

    Domain: [-5, 5]^D recommended
    Global minimum: 0
    """
    x = np.asarray(x)
    D = x.size
    df = 3  # degrees of freedom for student-t distribution

    # Use a fixed random state for reproducible centers
    rng = np.random.RandomState(seed)

    # Linear combination of wells
    total = 0.0
    max_depth = 0.0  # Track deepest well to offset function

    for i in range(m):
        # Set centers randomly in domain [-4, 4] (slightly inside [-5, 5])
        center = rng.uniform(-4.0, 4.0, size=D)

        # Different amplitudes for each well (increasing depth)
        amplitude = 0.5 + 1.5 * (i + 1) / m

        # Student-t distribution with negative coefficient to create a well
        dist_sq = np.sum((x - center)**2)
        well_value = -amplitude / (1.0 + dist_sq / df) ** ((df + D) / 2.0)
        total += well_value

    # Offset so global minimum is 0
    return total - max_depth


# helpers
def make_primitives(bounds, N):
    return [Grid(np.linspace(lo, hi, N, endpoint=True), [k]) for k, (lo, hi) in enumerate(bounds)]

def run_trc(func, bounds, D, N, r, nsweep, seed=0):
    np.random.seed(seed)
    primitives = make_primitives(bounds, N)
    #obj = Objective(func)
    obj = Objective(func, lambda x: -np.exp(-x))
    model = TensorRankOptimization(primitives, r)
    grid0 = random_grid_points(primitives, r, seed)
    grid = model.optimize(grid0, obj, nsweep)
    calls = obj.function_calls
    f_min = float(np.min(obj.logger.df['f'])) if len(obj.logger.df) else np.nan
    return calls, f_min, obj

def run_mt(func, bounds, D, N, r, nsweep, seed=0):
    np.random.seed(seed)
    primitives = make_primitives(bounds, N)
    #obj = Objective(func)
    obj = Objective(func, lambda x: -np.exp(-x))
    model = MatrixTrainOptimization(primitives, r)
    grid0 = random_grid_points(primitives, r, seed)
    grid = model.optimize(grid0, obj, nsweep)
    calls = obj.function_calls
    f_min = float(np.min(obj.logger.df['f'])) if len(obj.logger.df) else np.nan
    return calls, f_min, obj

def run_ttopt(func, bounds, D, N, r, nsweep, seed=0):
    np.random.seed(seed)
    f = len(bounds)
    #obj = Objective(func)
    obj = Objective(func, lambda x: np.exp(-x))
    G = tensor_train_graph(f, r, N)
    primitive_grid = [np.linspace(lo, hi, N, endpoint=True) for (lo, hi) in bounds]
    _ = ttnopt(G, obj, nsweep, primitive_grid)
    calls = obj.function_calls
    f_min = float(np.min(obj.logger.df['f'])) if len(obj.logger.df) else np.nan
    return calls, f_min, obj


# main
def compare_all(D, N, ranks, nsweep, seeds):
    tests = [
        # ("Ackley", ackley, [(-32.768, 32.768)]*D),
        # ("Alpine", alpine1, [(-10.0, 10.0)]*D),
        # ("Brown",  brown,  [(-1.0, 4.0)]*D),
        # ("Exponential", exp_neg_norm2, [(-1.0, 1.0)]*D),
        # ("Rosenbrock", rosenbrock, [(-0.0, 2.0)]*D),
        ("MultiWell", multi_well, [(-5.0, 5.0)]*D),
    ]
    rows = []
    for r in ranks:
        for seed in seeds:
            for name, f, bounds in tests:
                trc_calls, trc_min, obj_trc = run_trc(f, bounds, D, N, r, nsweep, seed)
                mt_calls,  mt_min,  obj_mt = run_mt(f, bounds, D, N, r, nsweep, seed)
                tt_calls,  tt_min,  obj_tt = run_ttopt(f, bounds, D, N, r, nsweep, seed)

                rows.append(
                    {
                        "Function": name,
                        "Method": "TRC",
                        "Rank": r,
                        "Seed": seed,
                        "Objective calls": trc_calls,
                        "best_f": trc_min
                    }
                )
                rows.append(
                    {
                        "Function": name,
                        "Method": "MatrixTrain",
                        "Rank": r,
                        "Seed": seed,
                        "Objective calls": mt_calls,
                        "best_f": mt_min
                    }
                )
                if tt_calls is not None:
                    rows.append(
                        {
                            "Function": name,
                            "Method": "TTOpt",
                            "Rank": r,
                            "Seed": seed,
                            "Objective calls": tt_calls,
                            "best_f": tt_min
                        }
                    )

            # if name == "Rosenbrock":
            #     print(obj_trc)
            #     print(obj_tt)

    df = pd.DataFrame(rows).sort_values(["Function", "Method", "Rank", "Seed"]).reset_index(drop=True)
    return df


df_results = compare_all(D=5, N=11, ranks=range(1, 6), nsweep=6, seeds=range(50))
df_results.to_csv("results.csv")
print(df_results)
# error vs num_calls (different ranks) #TODO
# linear sum assignment with multiple segments #TODO
