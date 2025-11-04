"""Top-level command line interface to run TRC/MT/TTOpt/... benchmarks and generate plots.

Example:
python benchmark.py \
--num_dimensions 7 --num_grid_points 5 --ranks 1 2 3 4 5 \
--num_sweeps 6 --seed 42 --num_experiments 10 \
--functions Ackley Alpine1 Brown Exponential Griewank Qing Rastrigin Schaffer \
--methods TRC MT TTOpt DA DE
"""

from __future__ import annotations
import argparse

from functions import FUNCTION_REGISTRY, F_OPT, get_tests, resolve_f_opt_map
from runner import compare_all, save_best_errors_csv
from plotting import make_plots


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Benchmark runner")
    p.add_argument(
        "--num_dimensions",type=int, default=3, help="Problem dimension"
    )
    p.add_argument(
        "--num_grid_points", type=int, default=10, help="Grid points per dimension"
    )
    p.add_argument(
        "--ranks", nargs="+", type=int, default=list(range(1, 5))
    )
    p.add_argument(
        "--num_sweeps", type=int, default=6, help="Number of sweeps"
    )
    p.add_argument(
        "--seed", nargs="+", type=int, default=list(range(10))
    )
    p.add_argument(
        "--num_experiments", type=int,
        help="Number of experiments to average over"
    )
    p.add_argument(
        "--methods", type=str, nargs="+", default=["TRC", "MT", "TTOpt"],
        help="Optimizers to run (TRC, MT, TTOpt)",
    )
    p.add_argument(
        "--functions", nargs="+",
        help=f"Functions to run (choices: {list(FUNCTION_REGISTRY.keys())})",
    )
    return p.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    tests = get_tests(args.num_dimensions, args.functions)
    df_results = compare_all(
        num_dimensions=args.num_dimensions,
        num_grid_points=args.num_grid_points,
        num_experiments=args.num_experiments,
        ranks=args.ranks,
        num_sweeps=args.num_sweeps,
        seed=args.seed,
        tests=tests,
        methods=args.methods,
    )
    df_results.to_csv("results.csv", index=False)
    print(df_results)
    print("Saved results")

    # dimension-aware f_opt map (fills Michalewicz for D=5 or D=10)
    f_opt_map = resolve_f_opt_map(F_OPT, args.num_dimensions)

    # Plots use the resolved map
    make_plots(df_results, f_opt_map)
    save_best_errors_csv(df_results, f_opt_map, path="best_errors.csv", sci_digits=1)


if __name__ == "__main__":
    main()
