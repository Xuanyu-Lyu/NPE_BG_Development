"""STEP 06 -- Core diagnostics for the fixed-N ACE NPEs.

This experiment trains one transient NPE for each training-budget/sample-size
cell and evaluates it using the same simulation and training conventions as
STEP 03.  The fitted NPE is kept only in memory and is discarded immediately
after its cell has been summarized.

Default design
--------------
K = 100,000, 300,000, 500,000 training simulations
N = 50, 100, 500, 1,000, 2,000, 5,000, 20,000 twin pairs
M = 1,000 shared test datasets per N
L = 2,000 posterior draws per test dataset

For each N, aggregation creates separate 3 x 3 figures (ACE parameters by K)
for SBC calibration ECDFs, recovery, and z-score versus contraction.  NRMSE
and R-squared are summarized as line plots over N.  Only compact diagnostic
tables and figures are retained; neither fitted models nor posterior draws are
serialized.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_model import ACE_PARAM_NAMES, RESULTS_DIR, resolve
from training_budget_utils import (
    evaluate_cell,
    resolve_device,
    seed_from,
    simulate_fixed_n,
    train_transient_posterior,
)


DEFAULT_K_VALUES = (100_000, 300_000, 500_000)
DEFAULT_N_VALUES = (50, 100, 500, 1_000, 2_000, 5_000, 20_000)
DEFAULT_OUTPUT_DIR = "step06_npe_diagnostics"
PRIOR_VARIANCE = 1.0 / 18.0  # Marginal variance under Dirichlet(1, 1, 1).
PRIOR_RANGE = 1.0
ECDF_REFERENCE_PROBABILITY = 0.99
K_COLORS = {
    100_000: "#4477AA",
    300_000: "#EE6677",
    500_000: "#228833",
}


def write_json_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def compact_number(value: int) -> str:
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}k"
    return str(value)


def cell_pairs(k_values: tuple[int, ...], n_values: tuple[int, ...]):
    """Return cells with K varying fastest within each N."""
    return tuple((k_value, n_value) for n_value in n_values for k_value in k_values)


def cell_stem(k_value: int, n_value: int) -> str:
    return f"K{k_value}_N{n_value}"


def run_cell(
    k_value: int,
    n_value: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Train, diagnose, summarize, and discard one transient NPE."""
    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    stem = cell_stem(k_value, n_value)
    results_path = cells_dir / f"{stem}.csv"
    metadata_path = cells_dir / f"{stem}.json"
    if results_path.exists() and metadata_path.exists() and not args.overwrite:
        print(f"Skipping completed diagnostic cell K={k_value:,}, N={n_value:,}")
        return

    (output_dir / "COMPLETE").unlink(missing_ok=True)
    started = time.perf_counter()
    print(f"Training transient NPE for diagnostics: K={k_value:,}, N={n_value:,}")

    training_seed = seed_from(args.seed, n_value, 10)
    features, ace = simulate_fixed_n(k_value, n_value, training_seed)
    posterior, scaler = train_transient_posterior(features, ace, args)

    # Identical N-specific test datasets are used for every K.
    test_seed = seed_from(args.seed, n_value, 20)
    test_features, test_ace = simulate_fixed_n(
        args.n_test_datasets, n_value, test_seed
    )
    posterior_seed = seed_from(args.seed, k_value, n_value, 30)
    results = evaluate_cell(
        posterior,
        scaler,
        test_features,
        test_ace,
        k_value,
        n_value,
        args.n_posterior_draws,
        posterior_seed,
    )
    results["posterior_z_score"] = np.divide(
        results["posterior_mean"] - results["truth"],
        results["posterior_sd"],
        out=np.full(len(results), np.nan, dtype=float),
        where=results["posterior_sd"].to_numpy(dtype=float) > 0.0,
    )
    results["posterior_contraction"] = (
        1.0 - results["posterior_sd"] ** 2 / PRIOR_VARIANCE
    )

    elapsed_seconds = time.perf_counter() - started
    metadata = {
        "K": k_value,
        "N": n_value,
        "H": 1,
        "M": args.n_test_datasets,
        "L": args.n_posterior_draws,
        "training_seed": args.training_seed,
        "simulation_seed": training_seed,
        "test_seed": test_seed,
        "posterior_seed": posterior_seed,
        "prior": "Dirichlet(1, 1, 1)",
        "prior_marginal_variance": PRIOR_VARIANCE,
        "elapsed_seconds": elapsed_seconds,
        "device": str(args.device_resolved),
        "models_saved": False,
        "posterior_draws_saved": False,
    }
    write_csv_atomic(results, results_path)
    write_json_atomic(metadata, metadata_path)
    print(
        f"Completed K={k_value:,}, N={n_value:,} in "
        f"{elapsed_seconds / 60.0:.1f} minutes -> {results_path}"
    )

    # Fitted model and full draws are deliberately never serialized.
    del posterior, scaler, features, ace, test_features, test_ace, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_metrics(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (k_value, n_value, parameter), group in results.groupby(
        ["K", "N", "parameter"], sort=True
    ):
        truth = group["truth"].to_numpy(dtype=float)
        estimate = group["posterior_mean"].to_numpy(dtype=float)
        error = estimate - truth
        residual_sum_squares = float(np.sum(error**2))
        total_sum_squares = float(np.sum((truth - truth.mean()) ** 2))
        r_squared = (
            1.0 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0.0
            else np.nan
        )
        rows.append(
            {
                "K": int(k_value),
                "N": int(n_value),
                "parameter": parameter,
                "M": len(group),
                "L": int(group["n_posterior_draws"].iloc[0])
                if "n_posterior_draws" in group
                else np.nan,
                "bias": float(error.mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "nrmse": float(np.sqrt(np.mean(error**2)) / PRIOR_RANGE),
                "r_squared": r_squared,
                "mean_posterior_sd": float(group["posterior_sd"].mean()),
                "mean_contraction": float(group["posterior_contraction"].mean()),
                "median_contraction": float(group["posterior_contraction"].median()),
                "mean_absolute_z_score": float(
                    group["posterior_z_score"].abs().mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def save_calibration_ecdf(
    results: pd.DataFrame,
    n_value: int,
    k_values: tuple[int, ...],
    n_draws: int,
    path: Path,
) -> None:
    """Plot SBC rank ECDF-minus-uniform with a simultaneous DKW band."""
    fig, axes = plt.subplots(
        len(ACE_PARAM_NAMES), len(k_values), figsize=(13.5, 11.0), squeeze=False
    )
    fig.suptitle(
        f"SBC calibration ECDF (N={n_value:,}; L={n_draws:,})", fontsize=15
    )
    alpha = 1.0 - ECDF_REFERENCE_PROBABILITY

    for row_index, parameter in enumerate(ACE_PARAM_NAMES):
        for column_index, k_value in enumerate(k_values):
            axis = axes[row_index, column_index]
            subset = results.loc[
                (results["N"] == n_value)
                & (results["K"] == k_value)
                & (results["parameter"] == parameter)
            ]
            ranks = subset["true_rank"].to_numpy(dtype=int)
            if len(ranks) == 0:
                raise ValueError(f"Missing K={k_value}, N={n_value}, {parameter}")
            if np.any((ranks < 0) | (ranks > n_draws)):
                raise ValueError("SBC ranks fall outside the expected [0, L] range")

            possible_ranks = np.arange(n_draws + 1)
            expected_cdf = (possible_ranks + 1) / (n_draws + 1)
            counts = np.bincount(ranks, minlength=n_draws + 1)
            observed_cdf = np.cumsum(counts) / len(ranks)
            difference = observed_cdf - expected_cdf
            epsilon = math.sqrt(math.log(2.0 / alpha) / (2.0 * len(ranks)))
            lower = np.maximum(0.0, expected_cdf - epsilon) - expected_cdf
            upper = np.minimum(1.0, expected_cdf + epsilon) - expected_cdf

            axis.fill_between(expected_cdf, lower, upper, color="0.88", linewidth=0)
            axis.plot(
                expected_cdf,
                difference,
                color=K_COLORS.get(k_value, "#4477AA"),
                linewidth=1.3,
            )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_xlim(0.0, 1.0)
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(f"K={compact_number(k_value)}")
            if column_index == 0:
                axis.set_ylabel(f"{parameter}\nEmpirical CDF - uniform CDF")
            if row_index == len(ACE_PARAM_NAMES) - 1:
                axis.set_xlabel("Uniform rank CDF")

    fig.text(
        0.5,
        0.01,
        "Gray region: 99% simultaneous Dvoretzky-Kiefer-Wolfowitz reference band",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_recovery(
    results: pd.DataFrame,
    metrics: pd.DataFrame,
    n_value: int,
    k_values: tuple[int, ...],
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(ACE_PARAM_NAMES), len(k_values), figsize=(13.5, 11.0), squeeze=False
    )
    fig.suptitle(f"Posterior-mean recovery (N={n_value:,} twin pairs)", fontsize=15)
    for row_index, parameter in enumerate(ACE_PARAM_NAMES):
        for column_index, k_value in enumerate(k_values):
            axis = axes[row_index, column_index]
            subset = results.loc[
                (results["N"] == n_value)
                & (results["K"] == k_value)
                & (results["parameter"] == parameter)
            ]
            metric = metrics.loc[
                (metrics["N"] == n_value)
                & (metrics["K"] == k_value)
                & (metrics["parameter"] == parameter)
            ].iloc[0]
            axis.scatter(
                subset["truth"],
                subset["posterior_mean"],
                s=9,
                alpha=0.28,
                color=K_COLORS.get(k_value, "#4477AA"),
                edgecolors="none",
            )
            axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
            axis.text(
                0.04,
                0.94,
                f"NRMSE={metric['nrmse']:.3f}\n$R^2$={metric['r_squared']:.3f}",
                transform=axis.transAxes,
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(f"K={compact_number(k_value)}")
            if column_index == 0:
                axis.set_ylabel(f"{parameter}\nPosterior mean")
            if row_index == len(ACE_PARAM_NAMES) - 1:
                axis.set_xlabel("True value")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_z_score_contraction(
    results: pd.DataFrame,
    n_value: int,
    k_values: tuple[int, ...],
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(ACE_PARAM_NAMES), len(k_values), figsize=(13.5, 11.0), squeeze=False
    )
    fig.suptitle(
        f"Posterior z-score and contraction (N={n_value:,} twin pairs)", fontsize=15
    )
    for row_index, parameter in enumerate(ACE_PARAM_NAMES):
        for column_index, k_value in enumerate(k_values):
            axis = axes[row_index, column_index]
            subset = results.loc[
                (results["N"] == n_value)
                & (results["K"] == k_value)
                & (results["parameter"] == parameter)
            ]
            axis.scatter(
                subset["posterior_contraction"],
                subset["posterior_z_score"],
                s=9,
                alpha=0.28,
                color=K_COLORS.get(k_value, "#4477AA"),
                edgecolors="none",
            )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.axhline(2.0, color="0.4", linestyle="--", linewidth=0.8)
            axis.axhline(-2.0, color="0.4", linestyle="--", linewidth=0.8)
            axis.axvline(0.0, color="0.6", linestyle=":", linewidth=0.8)
            z_scores = subset["posterior_z_score"].to_numpy(dtype=float)
            z_scores = z_scores[np.isfinite(z_scores)]
            n_above_two = int(np.sum(z_scores > 2.0))
            n_below_minus_two = int(np.sum(z_scores < -2.0))
            axis.text(
                0.03,
                0.97,
                f"z > 2: {n_above_two}\nz < -2: {n_below_minus_two}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.75",
                    "alpha": 0.85,
                    "boxstyle": "round,pad=0.25",
                },
            )
            axis.set_xlim(
                left=min(-0.1, float(subset["posterior_contraction"].min())),
                right=1.05,
            )
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(f"K={compact_number(k_value)}")
            if column_index == 0:
                axis.set_ylabel(f"{parameter}\nPosterior z-score")
            if row_index == len(ACE_PARAM_NAMES) - 1:
                axis.set_xlabel("Posterior contraction")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metric_over_n(
    metrics: pd.DataFrame,
    metric_name: str,
    y_label: str,
    k_values: tuple[int, ...],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(ACE_PARAM_NAMES), figsize=(14.0, 4.4), sharex=True)
    for axis, parameter in zip(axes, ACE_PARAM_NAMES):
        for k_value in k_values:
            subset = metrics.loc[
                (metrics["K"] == k_value) & (metrics["parameter"] == parameter)
            ].sort_values("N")
            axis.plot(
                subset["N"],
                subset[metric_name],
                marker="o",
                markersize=4.5,
                linewidth=1.6,
                color=K_COLORS.get(k_value),
                label=f"K={compact_number(k_value)}",
            )
        axis.set_xscale("log")
        axis.set_title(parameter)
        axis.set_xlabel("Twin-pair sample size N")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(y_label)
    axes[-1].legend(frameon=False)
    fig.suptitle(f"{y_label} across sample size and training budget", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def aggregate(args: argparse.Namespace, output_dir: Path) -> None:
    pairs = cell_pairs(args.k_values, args.n_values)
    cells_dir = output_dir / "cells"
    missing = []
    frames = []
    metadata_rows = []
    for k_value, n_value in pairs:
        stem = cell_stem(k_value, n_value)
        results_path = cells_dir / f"{stem}.csv"
        metadata_path = cells_dir / f"{stem}.json"
        if not results_path.exists() or not metadata_path.exists():
            missing.append(stem)
            continue
        frame = pd.read_csv(results_path)
        expected_rows = args.n_test_datasets * len(ACE_PARAM_NAMES)
        if len(frame) != expected_rows:
            raise ValueError(
                f"{results_path} has {len(frame)} rows; expected {expected_rows}"
            )
        with open(metadata_path) as handle:
            metadata = json.load(handle)
        expected_metadata = {
            "K": k_value,
            "N": n_value,
            "M": args.n_test_datasets,
            "L": args.n_posterior_draws,
        }
        mismatches = {
            key: (metadata.get(key), expected)
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"{metadata_path} is incompatible with this aggregation: {mismatches}"
            )
        required_columns = {
            "K",
            "N",
            "test_dataset_m",
            "parameter",
            "truth",
            "posterior_mean",
            "posterior_sd",
            "true_rank",
            "posterior_z_score",
            "posterior_contraction",
        }
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{results_path} is missing columns: {sorted(missing_columns)}"
            )
        frames.append(frame)
        metadata_rows.append(metadata)
    if missing:
        raise FileNotFoundError(
            "Cannot aggregate; missing diagnostic cells: " + ", ".join(missing)
        )

    results = pd.concat(frames, ignore_index=True)
    results["n_posterior_draws"] = args.n_posterior_draws
    metrics = compute_metrics(results)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    write_csv_atomic(results, output_dir / "diagnostic_results.csv")
    write_csv_atomic(metrics, output_dir / "diagnostic_metrics.csv")
    write_csv_atomic(pd.DataFrame(metadata_rows), output_dir / "cell_runtimes.csv")

    for n_value in args.n_values:
        save_calibration_ecdf(
            results,
            n_value,
            args.k_values,
            args.n_posterior_draws,
            figures_dir / f"calibration_ecdf_N{n_value}.png",
        )
        save_recovery(
            results,
            metrics,
            n_value,
            args.k_values,
            figures_dir / f"recovery_N{n_value}.png",
        )
        save_z_score_contraction(
            results,
            n_value,
            args.k_values,
            figures_dir / f"z_score_contraction_N{n_value}.png",
        )

    save_metric_over_n(
        metrics,
        "nrmse",
        "Normalized root mean squared error",
        args.k_values,
        figures_dir / "nrmse_by_n.png",
    )
    save_metric_over_n(
        metrics,
        "r_squared",
        r"$R^2$",
        args.k_values,
        figures_dir / "r_squared_by_n.png",
    )

    config = {
        "step": "06_npe_diagnostics",
        "K_values": list(args.k_values),
        "N_values": list(args.n_values),
        "H": 1,
        "M": args.n_test_datasets,
        "L": args.n_posterior_draws,
        "prior": "Dirichlet(1, 1, 1)",
        "prior_marginal_variance": PRIOR_VARIANCE,
        "nrmse_normalization_range": PRIOR_RANGE,
        "ecdf_reference_probability": ECDF_REFERENCE_PROBABILITY,
        "models_saved": False,
        "posterior_draws_saved": False,
        "training": {
            "validation_fraction": args.validation_fraction,
            "epochs": args.epochs,
            "stop_after_epochs": args.stop_after_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "flow_type": args.flow_type,
            "flow_hidden": args.flow_hidden,
            "flow_transforms": args.flow_transforms,
        },
    }
    write_json_atomic(config, output_dir / "config.json")
    (output_dir / "COMPLETE").write_text("complete\n")

    if not args.keep_cell_files:
        shutil.rmtree(cells_dir)
    print(f"Saved STEP 06 diagnostic tables and figures -> {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SBC, recovery, and contraction diagnostics for ACE NPEs"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cell-index", type=int, help="One-based K x N array index")
    mode.add_argument("--aggregate", action="store_true")
    mode.add_argument("--run-all", action="store_true", help="Local sequential run")
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--n-values", type=int, nargs="+", default=list(DEFAULT_N_VALUES))
    parser.add_argument("--n-test-datasets", type=int, default=1000, metavar="M")
    parser.add_argument("--n-posterior-draws", type=int, default=2000, metavar="L")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--stop-after-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument(
        "--flow-type", choices=("nsf", "maf", "maf_rqs", "mdn"), default="nsf"
    )
    parser.add_argument("--flow-hidden", type=int, default=64)
    parser.add_argument("--flow-transforms", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-cell-files", action="store_true")
    args = parser.parse_args()

    args.k_values = tuple(dict.fromkeys(args.k_values))
    args.n_values = tuple(dict.fromkeys(args.n_values))
    if any(value < 100 for value in args.k_values):
        parser.error("every K must be at least 100")
    if any(value < 2 for value in args.n_values):
        parser.error("every N must be at least 2")
    if args.n_test_datasets < 2:
        parser.error("M must be at least 2")
    if args.n_posterior_draws < 20:
        parser.error("L must be at least 20")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must lie between 0 and 0.5")
    if args.epochs < 1 or args.stop_after_epochs < 1 or args.batch_size < 1:
        parser.error("training counts must be positive")
    total_cells = len(args.k_values) * len(args.n_values)
    if args.cell_index is not None and not 1 <= args.cell_index <= total_cells:
        parser.error(f"--cell-index must lie between 1 and {total_cells}")
    args.device_resolved = resolve_device(args.device)
    return args


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir, RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = cell_pairs(args.k_values, args.n_values)
    print(
        f"STEP 06: H=1; M={args.n_test_datasets:,}; L={args.n_posterior_draws:,}; "
        f"cells={len(pairs)}"
    )
    if args.aggregate:
        aggregate(args, output_dir)
    elif args.run_all:
        for k_value, n_value in pairs:
            run_cell(k_value, n_value, args, output_dir)
        aggregate(args, output_dir)
    else:
        k_value, n_value = pairs[args.cell_index - 1]
        run_cell(k_value, n_value, args, output_dir)


if __name__ == "__main__":
    main()
