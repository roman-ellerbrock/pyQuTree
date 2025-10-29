"""Runner to compare TRC/MT/TTOpt/... optimizers on various benchmark functions."""

from __future__ import annotations
from typing import Iterable, List, Tuple, Callable
from tqdm import tqdm
import numpy as np
import pandas as pd
from numpy.random import SeedSequence
from helpers import run_trc, run_mt, run_ttopt


# A dictionary mapping optimizers' names to their corresponding runner functions
METHODS = {
    "TRC": run_trc,
    "MT": run_mt,
    "TTOpt": run_ttopt,
}


def compare_all(
    num_dimensions: int,
    num_grid_points: int,
    num_experiments: int,
    ranks: Iterable[int],
    num_sweeps: int,
    seed: int,
    tests: List[Tuple[str, Callable, list]],
    methods: Iterable[str],
) -> pd.DataFrame:
    """
    Run benchmarks across (ranks x experiments x test functions x optimizers).

    Seeds are produced with numpy.random.SeedSequence(seed).spawn(num_experiments),
    giving independent, reproducible child seeds. The same child seed is used for
    all methods within an experiment (Common Random Numbers).
    """
    rows = []
    ranks_list = list(ranks)
    methods_list = list(methods)

    ss = SeedSequence(seed)
    children = ss.spawn(num_experiments)
    seeds = [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]

    for rank in ranks_list:
        print(f"running rank = {rank}")
        for exp_idx, spawned_seed in tqdm(enumerate(seeds)):
            for name, f, bounds in tests:
                # sanity check: bounds length matches declared dimension
                if len(bounds) != num_dimensions:
                    raise ValueError(
                        f"Bounds length ({len(bounds)}) != num_dimensions ({num_dimensions}) for {name}"
                    )
                for method in methods_list:
                    if method not in METHODS:
                        raise KeyError(f"Unknown method: {method}")
                    calls, f_min, _ = METHODS[method](
                        f, bounds, num_grid_points, rank, num_sweeps, spawned_seed
                    )
                    rows.append(
                        {
                            "Function": name,
                            "Method": method,
                            "Rank": rank,
                            "Experiment": exp_idx,
                            "Seed": spawned_seed,
                            "Objective calls": calls,
                            "best_f": f_min,
                        }
                    )

    df = (
        pd.DataFrame(rows)
        .sort_values(["Function", "Method", "Rank", "Experiment", "Seed"])
        .reset_index(drop=True)
    )
    return df
