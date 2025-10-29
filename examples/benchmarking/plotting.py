""" Plotting functions for benchmark results. """

from __future__ import annotations
from typing import Dict, Optional, List
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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
    return {m: colors[i % len(colors)] for i, m in enumerate(sorted(methods))}


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
    y = best["error"].values
    ax.set_xlabel("Objective calls (best run)")
    ax.set_ylabel("Best error (best_f - f_opt)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Method")


def _grouped_boxplot(ax, df: pd.DataFrame, ycol: str, ranks: list[int], methods: list[str],
                     title: str, ylabel: str) -> None:
    colors = _method_colors(methods)

    M = len(methods)
    group_width = 0.80                 # total horizontal span reserved per rank
    box_w = group_width / max(M, 1)    # individual box width (tight, no overlap)

    positions, data, facecolors = [], [], []

    for r in ranks:
        center = r
        # left edge of the group; midpoint of the first box
        start = center - group_width/2 + box_w/2
        for j, m in enumerate(methods):
            pos = start + j*box_w
            vals = df[(df["Rank"] == r) & (df["Method"] == m)][ycol].dropna().values
            if vals.size == 0:
                vals = np.array([np.nan])
            positions.append(pos)
            data.append(vals)
            facecolors.append(colors[m])

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=box_w * 0.85,       # a touch narrower than the slot
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
        zorder=2,
    )

    for patch, c in zip(bp["boxes"], facecolors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.1)

    for key in ("medians", "whiskers", "caps"):
        for artist in bp[key]:
            artist.set_color("black")
            artist.set_linewidth(1.0)
            artist.set_zorder(3)

    # nice rank ticks centered at integers
    ax.set_xticks(ranks)
    ax.set_xlabel("Rank")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # legend
    handles = [Patch(facecolor=colors[m], edgecolor="black", alpha=0.55, label=m) for m in methods]
    ax.legend(handles=handles, title="Method")



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

        # 1) box: best function value by rank & method
        fig, ax = plt.subplots()
        _grouped_boxplot(
            ax, dff, ycol="best_f", ranks=ranks, methods=methods,
            title=f"{func}: Best function value vs rank (box over experiments)",
            ylabel="Best found f(x)",
        )
        _save(fig, _fname(outdir, func, plot_type="box_best_f_vs_rank", n_exp=n_exp))

        # 2) box: objective calls by rank & method
        fig, ax = plt.subplots()
        _grouped_boxplot(
            ax, dff, ycol="Objective calls", ranks=ranks, methods=methods,
            title=f"{func}: Objective calls vs rank (box over experiments)",
            ylabel="Objective calls",
        )
        _save(fig, _fname(outdir, func, plot_type="box_calls_vs_rank", n_exp=n_exp))

        # 3) box: best error by rank & method (if f_opt known)
        if pd.notna(f_opt_map.get(func, np.nan)):
            fig, ax = plt.subplots()
            _grouped_boxplot(
                ax, dff, ycol="error", ranks=ranks, methods=methods,
                title=f"{func}: Best error vs rank (box over experiments)",
                ylabel="Best error (best_f - f_opt)",
            )
            vals = dff["error"].dropna().values
            if vals.size > 0 and np.all(vals > 0) and (np.nanmax(vals) / np.nanmin(vals) > 10):
                ax.set_yscale("log")
            _save(fig, _fname(outdir, func, plot_type="box_error_vs_rank", n_exp=n_exp))

        # 4) Best-only summary: error vs objective calls (one point per method×rank)
        if pd.notna(f_opt_map.get(func, np.nan)):
            best = _agg_best_points(dff, methods=methods, ranks=ranks)
            if not best.empty:
                fig, ax = plt.subplots()
                _plot_best_error_vs_calls(ax, best, methods=methods)
                ax.set_title(
                    f"{func}: Best error vs calls (annotated by rank and method)"
                )
                _save(fig, _fname(outdir, func, plot_type="best_error_vs_calls", n_exp=n_exp))
