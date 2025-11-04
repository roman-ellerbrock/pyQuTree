"""Runner to compare TRC/MT/TTOpt/... optimizers on various benchmark functions."""

from __future__ import annotations
import re
from typing import Iterable, List, Tuple, Callable, Dict, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
from numpy.random import SeedSequence
from helpers import run_trc, run_mt, run_ttopt, run_de, run_da


# A dictionary mapping optimizers' names to their corresponding runner functions
METHODS = {
    "TRC": run_trc,
    "MT": run_mt,
    "TTOpt": run_ttopt,
    "DE": run_de,
    "DA": run_da,
}


# A dictionary mapping optimizers to their score transformation types
_METHOD_TRANSFORM = {
    "TRC":   "-exp",   # best_f = -exp(-f)
    "MT":    "-exp",
    "DE":    "-exp",   # ensure DE uses -exp(-f) in helpers
    "DA":    "-exp",   # ensure DA uses -exp(-f) in helpers
    "TTOpt": "+exp",   # best_f =  exp(-f)
}


def _invert_row_bestf_to_fx(best_f: float, method: str) -> float:
    """Invert logged score best_f back to the original objective f(x)."""
    if not np.isfinite(best_f):
        return np.nan
    eps = 1e-100
    t = _METHOD_TRANSFORM.get(method, None)
    if t == "-exp":
        # t in (-1,0]; f = -ln(-t)
        if best_f < 0:
            return float(-np.log(np.clip(-best_f, eps, 1.0)))
        # fall through to inference if sign is unexpected
    elif t == "+exp":
        # t in (0,1]; f = -ln(t)
        if 0 < best_f <= 1:
            return float(-np.log(np.clip(best_f, eps, 1.0)))
        # fall through

    # Fallback inference by sign/range:
    if best_f < 0:             # looks like -exp(-f)
        return float(-np.log(np.clip(-best_f, eps, 1.0)))
    if 0 < best_f <= 1:        # looks like +exp(-f)
        return float(-np.log(np.clip(best_f, eps, 1.0)))
    return float(best_f)       # assume already raw f


def _fmt_sci(x: float, digits: int = 1) -> str:
    """Format a float in scientific notation like 4.7e-6 (no zero-padded exponent)."""
    if not np.isfinite(x):
        return "nan"
    if x == 0.0:
        return "0"
    s = f"{float(x):.{digits}e}"              # e.g. '4.7e-06'
    return re.sub(r"e([+-])0*(\d+)$", r"e\1\2", s)  # -> '4.7e-6'


def _format_best_errors_table_for_csv(table: pd.DataFrame, digits: int = 1) -> pd.DataFrame:
    """Return a copy with numeric cells rendered in sci notation; 'None' left as-is."""
    out = table.copy()
    for col in out.columns:
        # Skip columns already forced to 'None' (dtype=object with all 'None')
        # Only format cells that parse as finite numbers
        vals = pd.to_numeric(out[col], errors="coerce")
        mask = vals.notna() & np.isfinite(vals)
        out.loc[mask, col] = vals[mask].astype(float).map(lambda v: _fmt_sci(v, digits))
    return out


def save_best_errors_csv(
    df_results: pd.DataFrame,
    f_opt_map: Dict[str, Optional[float]],
    path: str = "best_errors.csv",
    sci_digits: int = 1,   # 1 decimal -> like 4.7e-6; raise to 2 for 4.72e-6, etc.
) -> pd.DataFrame:
    """Build the best-errors table, write it as CSV with scientific-notation cells."""
    table = make_best_errors_table(df_results, f_opt_map)
    table_fmt = _format_best_errors_table_for_csv(table, digits=sci_digits)
    table_fmt.to_csv(path, index=True)
    return table


def make_best_errors_table(
    df_results: pd.DataFrame,
    f_opt_map: Dict[str, Optional[float]],
) -> pd.DataFrame:
    """
    Build (rows=Method, cols=Function) with best error across all ranks & experiments,
    computed on the ORIGINAL objective scale; unknown-min columns filled with 'None'.
    """
    df = df_results.copy()
    df["f_opt"] = df["Function"].map(f_opt_map)

    all_functions = sorted(df["Function"].unique())
    all_methods   = sorted(df["Method"].unique())

    known = df["f_opt"].notna() & np.isfinite(df["f_opt"])
    dfk = df[known].copy()

    if dfk.empty:
        table = pd.DataFrame(index=all_methods)
    else:
        # best_f is already the original f(x)
        dfk["error"] = (dfk["best_f"] - dfk["f_opt"]).clip(lower=0.0)
        best = dfk.groupby(["Method", "Function"], as_index=False)["error"].min()
        table = (
            best.pivot(index="Method", columns="Function", values="error")
                .reindex(index=all_methods)
        )

    for fn in all_functions:
        if fn not in getattr(table, "columns", []):
            table[fn] = np.nan
    table = table.reindex(columns=all_functions).astype(object)

    unknown_funcs = [
        fn for fn in all_functions
        if not (fn in f_opt_map and pd.notna(f_opt_map[fn]) and np.isfinite(f_opt_map[fn]))
    ]
    for fn in unknown_funcs:
        table[fn] = "None"

    return table


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
