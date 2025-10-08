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
def compare_all(D, N, r, nsweep, seeds):
    tests = [
        # ("Ackley", ackley, [(-32.768, 32.768)]*D),
        # ("Alpine", alpine1, [(-10.0, 10.0)]*D),
        # ("Brown",  brown,  [(-1.0, 4.0)]*D),
        # ("Exponential", exp_neg_norm2, [(-1.0, 1.0)]*D),
        ("Rosenbrock", rosenbrock, [(-0.0, 2.0)]*D),
    ]
    rows = []
    for seed in seeds:
        for name, f, bounds in tests:
            trc_calls, trc_min, obj_trc = run_trc(f, bounds, D, N, r, nsweep, seed)
            mt_calls,  mt_min,  obj_mt = run_mt(f, bounds, D, N, r, nsweep, seed)
            tt_calls,  tt_min,  obj_tt = run_ttopt(f, bounds, D, N, r, nsweep, seed)

            rows.append(
                {
                    "Function": name,
                    "Method": "TRC",
                    "Seed": seed,
                    "Rank": r,
                    "Objective calls": trc_calls,
                    "best f": trc_min
                }
            )
            rows.append(
                {
                    "Function": name,
                    "Method": "MatrixTrain",
                    "Seed": seed,
                    "Rank": r,
                    "Objective calls": mt_calls,
                    "best f": mt_min
                }
            )
            if tt_calls is not None:
                rows.append(
                    {
                        "Function": name,
                        "Method": "TTOpt",
                        "Seed": seed,
                        "Rank": r,
                        "Objective calls": tt_calls,
                        "best f": tt_min
                    }
                )

        # if name == "Rosenbrock":
        #     print(obj_trc)
        #     print(obj_tt)

    df = pd.DataFrame(rows).sort_values(["Function", "Method", "Seed"]).reset_index(drop=True)
    return df


df_results = compare_all(D=5, N=11, r=3, nsweep=6, seeds=range(10))
print(df_results)
# error vs num_calls (different ranks) #TODO
# linear sum assignment with multiple segments #TODO
