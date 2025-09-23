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


# helpers
def make_primitives(bounds, N):
    return [Grid(np.linspace(lo, hi, N, endpoint=True), [k]) for k, (lo, hi) in enumerate(bounds)]

def run_trc(func, bounds, D, N, r, nsweep, seed=0):
    np.random.seed(seed)
    primitives = make_primitives(bounds, N)
    obj = Objective(func)
    model = TensorRankOptimization(primitives, r)
    grid0 = random_grid_points(primitives, r)
    grid = model.optimize(grid0, obj, nsweep)
    calls = obj.function_calls
    f_min = float(np.min(obj.logger.df['f'])) if len(obj.logger.df) else np.nan
    return calls, f_min, obj

def run_mt(func, bounds, D, N, r, nsweep, seed=0):
    np.random.seed(seed)
    primitives = make_primitives(bounds, N)
    obj = Objective(func)
    model = MatrixTrainOptimization(primitives, r)
    grid0 = random_grid_points(primitives, r)
    grid = model.optimize(grid0, obj, nsweep)
    calls = obj.function_calls
    f_min = float(np.min(obj.logger.df['f'])) if len(obj.logger.df) else np.nan
    return calls, f_min, obj

def run_ttopt(func, bounds, D, N, r, nsweep, seed=0):
    np.random.seed(seed)
    f = len(bounds)
    obj = Objective(func)
    G = tensor_train_graph(f, r, N)
    primitive_grid = [np.linspace(lo, hi, N, endpoint=True) for (lo, hi) in bounds]
    _ = ttnopt(G, obj, nsweep, primitive_grid)
    calls = obj.function_calls
    f_min = float(np.min(obj.logger.df['f'])) if len(obj.logger.df) else np.nan
    return calls, f_min, obj


# main
def compare_all(D, N, r, nsweep, seed):
    tests = [
        ("Ackley", ackley, [(-32.768, 32.768)]*D),
        ("Alpine", alpine1, [(-10.0, 10.0)]*D),
        ("Brown",  brown,  [(-1.0, 4.0)]*D),
        ("Exponential", exp_neg_norm2, [(-1.0, 1.0)]*D),
    ]
    rows = []
    for name, f, bounds in tests:
        trc_calls, trc_min, _ = run_trc(f, bounds, D, N, r, nsweep, seed)
        mt_calls,  mt_min,  _ = run_mt(f, bounds, D, N, r, nsweep, seed)
        tt_calls,  tt_min,  _ = run_ttopt(f, bounds, D, N, r, nsweep, seed)

        rows.append(
            {
                "Function": name,
                "Method": "TRC",
                "Objective calls": trc_calls,
                "best f": trc_min
            }
        )
        rows.append(
            {
                "Function": name,
                "Method": "MatrixTrain",
                "Objective calls": mt_calls,
                "best f": mt_min
            }
        )
        if tt_calls is not None:
            rows.append(
                {
                    "Function": name,
                    "Method": "TTOpt",
                    "Objective calls": tt_calls,
                    "best f": tt_min
                }
            )

    df = pd.DataFrame(rows).sort_values(["Function", "Method"]).reset_index(drop=True)
    return df


df_results = compare_all(D=5, N=11, r=8, nsweep=6, seed=0)
print(df_results)
