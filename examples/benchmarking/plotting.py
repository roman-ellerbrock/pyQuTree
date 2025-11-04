""" Plotting functions for benchmark results. """

from __future__ import annotations
from typing import Dict, Optional, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved {path}")


def _method_colors(methods: List[str]) -> Dict[str, str]:
    cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = (cycle.by_key().get("color", []) if cycle else [])
    if not colors:
        colors = [f"C{i}" for i in range(10)]
    # preserve the incoming order
    return {m: colors[i % len(colors)] for i, m in enumerate(methods)}


def _slug(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_")).strip("_")


def _fname(outdir: str, func: str, plot_type: str, n_exp: int) -> str:
    return os.path.join(
        outdir,
        f"func_{_slug(func)}_type_{plot_type}_num_experiments_{n_exp}.png"
    )


def _agg_best_points(dff: pd.DataFrame, methods: list[str], ranks: list[int],
                     tol: float = 1e-12) -> pd.DataFrame:
    rows = []
    for m in methods:
        for r in ranks:
            g = dff[(dff["Method"] == m) & (dff["Rank"] == r) & dff["error"].notna()]
            if g.empty:
                continue
            err_min = float(g["error"].min())
            cand = g[np.isclose(g["error"].values, err_min, rtol=1e-6, atol=tol)]
            # tie-break by minimal calls
            idx = cand["Objective calls"].idxmin()
            rows.append({
                "Method": m,
                "Rank": int(r),
                "calls": float(g.loc[idx, "Objective calls"]),
                "error": float(g.loc[idx, "error"]),
            })
    return pd.DataFrame(rows)


def _plot_best_error_vs_calls(ax, best: pd.DataFrame, methods: list[str]) -> None:
    colors = _method_colors(methods)
    for m in methods:
        sm = best[best["Method"] == m].sort_values("Rank")
        if sm.empty:
            continue
        ax.plot(sm["calls"], sm["error"], "-o", label=m, color=colors[m], alpha=0.95)
        # annotate with rank
        for _, row in sm.iterrows():
            ax.annotate(str(int(row["Rank"])),
                        (row["calls"], row["error"]),
                        textcoords="offset points", xytext=(5, 0),
                        fontsize=8, color=colors[m])
    ax.set_xscale("log")
    ax.set_xlabel("Objective calls (best run)")
    ax.set_ylabel("Best error (best_f - f_opt)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Method")


def _scatter_with_sem(ax, df: pd.DataFrame, ycol: str,
                   ranks: list[int], methods: list[str],
                   title: str, ylabel: str,
                   logy: bool = False,
                   offset: float = 0.12) -> None:
    """
    Plot mean ± SEM across experiments for each (Method, Rank) as jittered dots with error bars.
    """
    # aggregate over experiments
    g = (
        df.groupby(["Method", "Rank"])[ycol]
          .agg(n="count",
               mean="mean",
               sem=lambda s: s.std(ddof=1) / np.sqrt(max(len(s), 1)))
          .reset_index()
    )

    # consistent order & colors
    methods_order = [m for m in methods if m in set(g["Method"])]
    ranks_sorted = sorted(ranks)
    colors = _method_colors(methods_order)

    M = len(methods_order)
    for j, m in enumerate(methods_order):
        gi = g[g["Method"] == m].set_index("Rank").reindex(ranks_sorted)
        y = gi["mean"].to_numpy()
        yerr = gi["sem"].fillna(0.0).to_numpy()
        x = np.asarray(ranks_sorted, dtype=float) + (j - (M - 1) / 2.0) * offset

        ax.errorbar(
            x, y, yerr=yerr,
            fmt="-o", lw=1.0, ms=4, capsize=2,
            color=colors[m], ecolor="black", alpha=0.9, label=m
        )

    ax.set_xlabel("Rank")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(ranks_sorted)
    if logy:
        # guard against non-positive values on log scale
        ymin = np.nanmin(np.where(yerr > 0, y - yerr, y))
        if np.isfinite(ymin) and ymin > 0:
            ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Method")


def make_plots(df_results: pd.DataFrame, f_opt_map: Dict[str, Optional[float]], outdir: str = "plots") -> None:
    # sanity check
    required = {"Function", "Method", "Rank", "best_f", "Objective calls"}
    missing = required - set(df_results.columns)
    if missing:
        raise ValueError(f"make_plots: missing columns: {sorted(missing)}")

    df = df_results.copy()
    df["f_opt"] = df["Function"].map(f_opt_map)
    df["error"] = np.where(df["f_opt"].notna(), np.maximum(df["best_f"] - df["f_opt"], 0.0), np.nan)

    functions = sorted(df["Function"].unique())

    for func in functions:
        dff = df[df["Function"] == func].copy()
        methods = sorted(dff["Method"].unique())
        ranks = sorted(dff["Rank"].unique().tolist())
        # infer number of experiments per function
        n_exp = int(dff["Experiment"].nunique() if "Experiment" in dff.columns
                    else dff.groupby(["Method", "Rank"]).size().max())

        # 1) dots: best function value by rank & method
        fig, ax = plt.subplots()
        _scatter_with_sem(
            ax, dff, ycol="best_f", ranks=ranks, methods=methods,
            title=f"{func}: Best function value vs rank (mean ± SEM)",
            ylabel="Best found f(x)",
            logy=False,
        )
        _save(fig, _fname(outdir, func, plot_type="dots_best_f_vs_rank", n_exp=n_exp))

        # 2) dots: objective calls by rank & method (already dots)
        fig, ax = plt.subplots()
        _scatter_with_sem(
            ax, dff, ycol="Objective calls", ranks=ranks, methods=methods,
            title=f"{func}: Objective calls vs rank (mean ± SEM)",
            ylabel="Objective calls (mean ± SEM across experiments)",
            logy=False,
        )
        _save(fig, _fname(outdir, func, plot_type="dots_calls_vs_rank", n_exp=n_exp))

        # 3) dots: best error by rank & method (if f_opt known)
        if pd.notna(f_opt_map.get(func, np.nan)):
            fig, ax = plt.subplots()
            # log y if dynamic range is wide and positive
            use_logy = False
            vals = dff["error"].dropna().values
            if vals.size > 0:
                vmin, vmax = np.nanmin(vals), np.nanmax(vals)
                use_logy = (vmin > 0) and (vmax / max(vmin, 1e-300) > 10)
            _scatter_with_sem(
                ax, dff, ycol="error", ranks=ranks, methods=methods,
                title=f"{func}: Best error vs rank (mean ± SEM)",
                ylabel="Best error (best_f - f_opt)",
                logy=use_logy,
            )
            _save(fig, _fname(outdir, func, plot_type="dots_error_vs_rank", n_exp=n_exp))

        # 4) Best-only summary: error vs objective calls (one point per method×rank)
        if pd.notna(f_opt_map.get(func, np.nan)):
            best = _agg_best_points(dff, methods=methods, ranks=ranks)
            if not best.empty:
                fig, ax = plt.subplots()
                _plot_best_error_vs_calls(ax, best, methods=methods)
                ax.set_title(f"{func}: Best error vs calls (annotated by rank and method)")
                _save(fig, _fname(outdir, func, plot_type="best_error_vs_calls", n_exp=n_exp))
