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
for marginal SBC calibration ECDFs, recovery, and z-score versus contraction.
An additional covariance-prediction RMSE SBC figure compares the K values for
each N, using the same joint posterior draws and original covariance features.
NRMSE and R-squared are summarized as line plots over N. Posterior corner
plots and noisy covariance-summary PPCs inspect one selected dataset per N,
shared across K. Compact PPC summaries cover all test datasets. Joint draws
and PPC replicates are retained only for the selected dataset in each cell;
fitted models are never serialized.
Aggregation and replotting also compute log-gamma scores from retained SBC
ranks and annotate marginal/RMSE ECDF panels without changing their bands.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_model import ACE_PARAM_NAMES, COV_FEATURE_NAMES, RESULTS_DIR, resolve
from rmse_sbc import RMSE_SBC_DEFINITION, summarize_rmse_sbc
from sbc_log_gamma import LOG_GAMMA_SETTINGS, summarize_log_gamma
from posterior_checks import save_corner_plot
from posterior_predictive_checks import (
    PPC_DEFINITION, posterior_predictive_replicates, summarize_ppc, save_ppc_plots,
)
from training_budget_utils import (
    evaluate_cell,
    resolve_device,
    seed_from,
    simulate_fixed_n,
    train_transient_posterior,
)


DEFAULT_K_VALUES = (100_000, 300_000, 500_000)
DEFAULT_N_VALUES = (50, 100, 500, 1_000, 2_000, 5_000, 20_000)
DEFAULT_OUTPUT_DIR = "step06_npe_diagnostics_ppc"
PRIOR_VARIANCE = 1.0 / 18.0  # Marginal variance under Dirichlet(1, 1, 1).
PRIOR_RANGE = 1.0
ECDF_REFERENCE_PROBABILITY = 0.95
ECDF_BAND_SIMULATIONS = 1_000
ECDF_BAND_SEED = 20_260_306
ECDF_BAND_MAX_POINTS = 1_000
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


def training_settings(args: argparse.Namespace) -> dict:
    return {
        name: getattr(args, name)
        for name in (
            "validation_fraction", "epochs", "stop_after_epochs", "batch_size",
            "learning_rate", "flow_type", "flow_hidden", "flow_transforms",
        )
    }


def expected_cell_metadata(k_value: int, n_value: int, args: argparse.Namespace) -> dict:
    return {
        "K": k_value,
        "N": n_value,
        "H": 1,
        "M": args.n_test_datasets,
        "L": args.n_posterior_draws,
        "training_seed": args.training_seed,
        "simulation_seed": seed_from(args.seed, n_value, 10),
        "test_seed": seed_from(args.seed, n_value, 20),
        "posterior_seed": seed_from(args.seed, k_value, n_value, 30),
        "rmse_tie_seed": seed_from(args.seed, k_value, n_value, 40),
        "rmse_sbc": RMSE_SBC_DEFINITION,
        "ppc": PPC_DEFINITION,
        "inspection_dataset": args.inspection_dataset,
        "n_ppc_replicates": args.n_ppc_replicates,
        "ppc_seed": seed_from(args.seed, k_value, n_value, 50),
        "training": training_settings(args),
    }


def validate_cell_metadata(metadata: dict, expected: dict, path: Path) -> None:
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"{path} is incompatible with this run: {mismatches}. "
            "Use a fresh --output-dir or rerun the cell with --overwrite."
        )


def validate_rmse_results(
    frame: pd.DataFrame, k_value: int, n_value: int, n_tests: int, n_draws: int
) -> None:
    required = {
        "K", "N", "test_dataset_m", "true_rmse", "posterior_rmse_mean",
        "posterior_rmse_sd", "rank_strict", "rank_ties", "true_rank",
        "true_rank_percentile", "n_posterior_draws",
    }
    if required.difference(frame.columns):
        raise ValueError(f"RMSE SBC is missing columns: {sorted(required - set(frame.columns))}")
    if len(frame) != n_tests or set(frame["test_dataset_m"]) != set(range(1, n_tests + 1)):
        raise ValueError(f"RMSE SBC K={k_value}, N={n_value} needs one row per dataset")
    if not (
        (frame["K"] == k_value).all()
        and (frame["N"] == n_value).all()
        and (frame["n_posterior_draws"] == n_draws).all()
    ):
        raise ValueError("RMSE SBC cell settings do not match the requested run")
    numeric = frame[list(required - {"K", "N", "test_dataset_m"})].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("RMSE SBC contains non-finite values")
    ranks = frame["true_rank"].to_numpy(dtype=float)
    strict = frame["rank_strict"].to_numpy(dtype=float)
    ties = frame["rank_ties"].to_numpy(dtype=float)
    if (
        np.any(ranks != np.floor(ranks))
        or np.any(strict != np.floor(strict))
        or np.any(ties != np.floor(ties))
        or np.any(strict < 0) or np.any(ties < 0)
        or np.any(strict + ties > n_draws)
        or np.any(ranks < strict) or np.any(ranks > strict + ties)
        or not np.allclose(frame["true_rank_percentile"], ranks / n_draws)
    ):
        raise ValueError("Invalid RMSE SBC ranks or tie counts")


def selected_samples_path(output_dir: Path, k_value: int, n_value: int) -> Path:
    return output_dir / "diagnostic_samples" / f"{cell_stem(k_value, n_value)}.npz"


def validate_ppc_results(
    frame: pd.DataFrame, k_value: int, n_value: int, n_tests: int, n_replicates: int
) -> None:
    required = {"K", "N", "test_dataset_m", "feature", "observed", "predictive_mean",
                "predictive_sd", "predictive_q2_5", "predictive_median", "predictive_q97_5",
                "upper_tail_fraction", "observed_in_95_interval", "n_replicates"}
    if required.difference(frame.columns):
        raise ValueError(f"PPC table is missing columns: {sorted(required - set(frame.columns))}")
    if (len(frame) != 4 * n_tests or frame.duplicated(["test_dataset_m", "feature"]).any()
            or set(frame["feature"]) != set(COV_FEATURE_NAMES)
            or not (frame["K"] == k_value).all() or not (frame["N"] == n_value).all()
            or not (frame["n_replicates"] == n_replicates).all()):
        raise ValueError("PPC rows do not match the requested cell")
    for feature in COV_FEATURE_NAMES:
        if set(frame.loc[frame["feature"] == feature, "test_dataset_m"]) != set(range(1, n_tests + 1)):
            raise ValueError("PPC table has missing dataset IDs")
    if not np.isfinite(frame[list(required - {"feature"})].to_numpy(dtype=float)).all():
        raise ValueError("PPC table contains non-finite values")
    if (not frame["upper_tail_fraction"].between(0, 1).all()
            or not frame["observed_in_95_interval"].isin([0, 1]).all()
            or (frame["predictive_sd"] < 0).any()
            or (frame["predictive_q2_5"] > frame["predictive_median"]).any()
            or (frame["predictive_median"] > frame["predictive_q97_5"]).any()):
        raise ValueError("Invalid PPC summary values")


def load_selected_samples(
    output_dir: Path, k_value: int, n_value: int, args: argparse.Namespace
) -> dict[str, np.ndarray]:
    path = selected_samples_path(output_dir, k_value, n_value)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    expected_shapes = {"posterior": (args.n_posterior_draws, 3), "truth": (3,),
                       "observed": (4,), "replicates": (args.n_ppc_replicates, 4)}
    for key, shape in expected_shapes.items():
        if key not in data or data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid selected-dataset {key} in {path}")
    for key, value in {"K": k_value, "N": n_value, "test_dataset_m": args.inspection_dataset}.items():
        if key not in data or data[key].shape != () or data[key].item() != value:
            raise ValueError(f"Selected-dataset {key} does not match this run: {path}")
    return data


def save_inspection_figures(args: argparse.Namespace, output_dir: Path) -> None:
    for k_value, n_value in cell_pairs(args.k_values, args.n_values):
        data = load_selected_samples(output_dir, k_value, n_value, args)
        stem = f"{cell_stem(k_value, n_value)}_dataset{args.inspection_dataset}"
        title = f"K={k_value:,}; N={n_value:,}; test dataset {args.inspection_dataset}"
        save_corner_plot(data["posterior"], data["truth"],
                         output_dir / "figures" / "posterior" / f"corner_{stem}.png", title)
        save_ppc_plots(data["replicates"], data["observed"],
                       output_dir / "figures" / "ppc" / f"ppc_{stem}.png", title)


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
    rmse_path = cells_dir / f"{stem}_rmse_sbc.csv"
    ppc_path = cells_dir / f"{stem}_ppc.csv"
    samples_path = selected_samples_path(output_dir, k_value, n_value)
    metadata_path = cells_dir / f"{stem}.json"
    expected_metadata = expected_cell_metadata(k_value, n_value, args)
    if metadata_path.exists() and not args.overwrite:
        with metadata_path.open() as handle:
            validate_cell_metadata(json.load(handle), expected_metadata, metadata_path)
    if all(path.exists() for path in (results_path, metadata_path, rmse_path, ppc_path, samples_path)) and not args.overwrite:
        validate_rmse_results(
            pd.read_csv(rmse_path), k_value, n_value,
            args.n_test_datasets, args.n_posterior_draws,
        )
        validate_ppc_results(pd.read_csv(ppc_path), k_value, n_value,
                             args.n_test_datasets, args.n_ppc_replicates)
        load_selected_samples(output_dir, k_value, n_value, args)
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
    rmse_tie_rng = np.random.default_rng(expected_metadata["rmse_tie_seed"])
    rmse_rows = []
    ppc_rows = []
    selected = {}

    def collect_diagnostics(test_index: int, draws: np.ndarray) -> None:
        observed = test_features[test_index - 1]
        truth = test_ace[test_index - 1]
        rmse_rows.append({
            "K": k_value,
            "N": n_value,
            "model_index_h": 1,
            "test_dataset_m": test_index,
            **dict(zip(COV_FEATURE_NAMES, map(float, observed))),
            **dict(zip((f"true_{name}" for name in ACE_PARAM_NAMES), map(float, truth))),
            **summarize_rmse_sbc(draws, truth, observed, rmse_tie_rng),
        })
        replicates = posterior_predictive_replicates(
            draws, n_value, args.n_ppc_replicates,
            seed_from(expected_metadata["ppc_seed"], test_index),
        )
        ppc_rows.extend({"K": k_value, "N": n_value, "test_dataset_m": test_index, **row}
                        for row in summarize_ppc(replicates, observed))
        if test_index == args.inspection_dataset:
            selected.update(posterior=draws.copy(), truth=truth.copy(), observed=observed.copy(),
                            replicates=replicates, K=np.array(k_value), N=np.array(n_value),
                            test_dataset_m=np.array(test_index))

    results = evaluate_cell(
        posterior,
        scaler,
        test_features,
        test_ace,
        k_value,
        n_value,
        args.n_posterior_draws,
        posterior_seed,
        draw_callback=collect_diagnostics,
    )
    ppc_results = pd.DataFrame(ppc_rows)
    validate_ppc_results(ppc_results, k_value, n_value, args.n_test_datasets, args.n_ppc_replicates)
    rmse_results = pd.DataFrame(rmse_rows)
    validate_rmse_results(
        rmse_results, k_value, n_value, args.n_test_datasets, args.n_posterior_draws
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
        **expected_metadata,
        "prior": "Dirichlet(1, 1, 1)",
        "prior_marginal_variance": PRIOR_VARIANCE,
        "elapsed_seconds": elapsed_seconds,
        "device": str(args.device_resolved),
        "models_saved": False,
        "posterior_draws_saved": "selected dataset only",
    }
    write_csv_atomic(results, results_path)
    write_csv_atomic(rmse_results, rmse_path)
    write_csv_atomic(ppc_results, ppc_path)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = samples_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **selected)
    temporary.replace(samples_path)
    write_json_atomic(metadata, metadata_path)
    print(
        f"Completed K={k_value:,}, N={n_value:,} in "
        f"{elapsed_seconds / 60.0:.1f} minutes -> {results_path}"
    )

    # Only the selected dataset's draws survive; the fitted model is discarded.
    del posterior, scaler, features, ace, test_features, test_ace, results, rmse_results, rmse_rows
    del ppc_results, ppc_rows, selected
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


def compute_log_gamma_metrics(
    results: pd.DataFrame, n_draws: int, rmse_results: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Summarize every cell/quantity using all its existing SBC ranks."""
    columns = ["K", "N", "parameter", "test_dataset_m", "true_rank"]
    rank_tables = [results[columns]]
    if rmse_results is not None:
        rank_tables.append(rmse_results.assign(parameter="RMSE")[columns])
    ranks = pd.concat(rank_tables, ignore_index=True)
    if ranks.duplicated(["K", "N", "parameter", "test_dataset_m"]).any():
        raise ValueError("Log gamma requires one rank per dataset, cell, and quantity")
    rows = []
    for (k_value, n_value, quantity), group in ranks.groupby(["K", "N", "parameter"], sort=True):
        rows.append({
            "K": int(k_value), "N": int(n_value), "quantity": quantity,
            **summarize_log_gamma(group["true_rank"].to_numpy(), n_draws),
        })
    return pd.DataFrame(rows)


def simultaneous_ecdf_band(
    n_estimates: int,
    confidence: float = ECDF_REFERENCE_PROBABILITY,
    n_simulations: int = ECDF_BAND_SIMULATIONS,
    seed: int = ECDF_BAND_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return BayesFlow-style simultaneous uniform-ECDF difference bands.

    This follows the simulation and binomial-quantile construction of
    Saeilynoja, Buerkner, and Vehtari (2022), as used by BayesFlow's
    ``simultaneous_ecdf_bands`` utility.
    """
    if n_estimates < 2:
        raise ValueError("At least two estimates are required for an ECDF band")
    if not 0.0 < confidence < 1.0:
        raise ValueError("ECDF band confidence must lie strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    z = np.linspace(1e-5, 1.0 - 1e-5, min(n_estimates, ECDF_BAND_MAX_POINTS))
    gammas = np.empty(n_simulations, dtype=float)

    # Sorting plus searchsorted is equivalent to BayesFlow's broadcasted count,
    # but avoids constructing a potentially gigabyte-sized M x K x N array.
    for simulation_index in range(n_simulations):
        uniform_order_stats = np.sort(rng.random(n_estimates))
        empirical_counts = np.searchsorted(uniform_order_stats, z, side="right")
        lower_tail = binom.cdf(empirical_counts, n_estimates, z)
        upper_tail = 1.0 - binom.cdf(empirical_counts - 1, n_estimates, z)
        gammas[simulation_index] = 2.0 * np.min(
            np.minimum(lower_tail, upper_tail)
        )

    simultaneous_alpha = 1.0 - confidence
    pointwise_alpha = float(
        np.percentile(gammas, 100.0 * simultaneous_alpha)
    )
    lower_cdf = binom.ppf(pointwise_alpha / 2.0, n_estimates, z) / n_estimates
    upper_cdf = (
        binom.ppf(1.0 - pointwise_alpha / 2.0, n_estimates, z) / n_estimates
    )
    return z, lower_cdf - z, upper_cdf - z


def save_calibration_ecdf(
    results: pd.DataFrame,
    n_value: int,
    k_values: tuple[int, ...],
    n_draws: int,
    path: Path,
    *,
    variable_names: tuple[str, ...] = tuple(ACE_PARAM_NAMES),
    title: str = "SBC calibration ECDF",
    log_gamma_results: pd.DataFrame | None = None,
) -> None:
    """Plot SBC rank ECDF-minus-uniform with a tapered simultaneous band."""
    fig, axes = plt.subplots(
        len(variable_names), len(k_values),
        figsize=(13.5, 11.0 if len(variable_names) == 3 else 4.5), squeeze=False
    )
    fig.suptitle(
        f"{title} (N={n_value:,}; L={n_draws:,})", fontsize=15
    )
    reference_bands: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for row_index, parameter in enumerate(variable_names):
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
            if len(ranks) not in reference_bands:
                reference_bands[len(ranks)] = simultaneous_ecdf_band(len(ranks))
            band_x, lower, upper = reference_bands[len(ranks)]

            axis.fill_between(band_x, lower, upper, color="0.88", linewidth=0)
            axis.plot(
                expected_cdf,
                difference,
                color=K_COLORS.get(k_value, "#4477AA"),
                linewidth=1.3,
            )
            axis.axhline(0.0, color="black", linewidth=0.8)
            if log_gamma_results is not None:
                score = log_gamma_results.loc[
                    (log_gamma_results["K"] == k_value)
                    & (log_gamma_results["N"] == n_value)
                    & (log_gamma_results["quantity"] == parameter), "log_gamma"
                ]
                if len(score) != 1:
                    raise ValueError(f"Expected one log-gamma score for K={k_value}, N={n_value}, {parameter}")
                axis.text(0.98, 0.97, f"Log gamma = {float(score.iloc[0]):+.2f}",
                          transform=axis.transAxes, ha="right", va="top", fontsize=9,
                          bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
            axis.set_xlim(0.0, 1.0)
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(f"K={compact_number(k_value)}")
            if column_index == 0:
                axis.set_ylabel(f"{parameter}\nEmpirical CDF - uniform CDF")
            if row_index == len(variable_names) - 1:
                axis.set_xlabel("Uniform rank CDF")

    fig.text(
        0.5,
        0.01,
        "Gray region: 95% simulation-calibrated simultaneous ECDF reference band",
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


def save_all_figures(
    results: pd.DataFrame,
    metrics: pd.DataFrame,
    args: argparse.Namespace,
    figures_dir: Path,
    rmse_results: pd.DataFrame | None = None,
    log_gamma_results: pd.DataFrame | None = None,
) -> None:
    """Create every diagnostic figure from retained tabular results."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    for n_value in args.n_values:
        save_calibration_ecdf(
            results,
            n_value,
            args.k_values,
            args.n_posterior_draws,
            figures_dir / f"calibration_ecdf_N{n_value}.png",
            log_gamma_results=log_gamma_results,
        )
        if rmse_results is not None:
            save_calibration_ecdf(
                rmse_results.assign(parameter="RMSE"),
                n_value,
                args.k_values,
                args.n_posterior_draws,
                figures_dir / f"predictive_rmse_sbc_ecdf_N{n_value}.png",
                variable_names=("RMSE",),
                title="Covariance-prediction RMSE SBC",
                log_gamma_results=log_gamma_results,
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


def aggregate(args: argparse.Namespace, output_dir: Path) -> None:
    (output_dir / "COMPLETE").unlink(missing_ok=True)
    pairs = cell_pairs(args.k_values, args.n_values)
    cells_dir = output_dir / "cells"
    missing = []
    frames = []
    rmse_frames = []
    ppc_frames = []
    metadata_rows = []
    for k_value, n_value in pairs:
        stem = cell_stem(k_value, n_value)
        results_path = cells_dir / f"{stem}.csv"
        rmse_path = cells_dir / f"{stem}_rmse_sbc.csv"
        metadata_path = cells_dir / f"{stem}.json"
        ppc_path = cells_dir / f"{stem}_ppc.csv"
        if not results_path.exists() or not metadata_path.exists() or not rmse_path.exists():
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
        validate_cell_metadata(
            metadata, expected_cell_metadata(k_value, n_value, args), metadata_path
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
        if (
            not (frame["K"] == k_value).all() or not (frame["N"] == n_value).all()
            or frame.duplicated(["test_dataset_m", "parameter"]).any()
            or set(frame["parameter"]) != set(ACE_PARAM_NAMES)
        ):
            raise ValueError(f"{results_path} has incorrect or duplicate marginal SBC rows")
        for parameter in ACE_PARAM_NAMES:
            if set(frame.loc[frame["parameter"] == parameter, "test_dataset_m"]) != set(
                range(1, args.n_test_datasets + 1)
            ):
                raise ValueError(f"{results_path} has missing marginal SBC dataset IDs")
        rmse_frame = pd.read_csv(rmse_path)
        validate_rmse_results(
            rmse_frame, k_value, n_value, args.n_test_datasets, args.n_posterior_draws
        )
        if not ppc_path.exists() or not selected_samples_path(output_dir, k_value, n_value).exists():
            raise FileNotFoundError(f"Missing PPC or selected-dataset samples for {stem}; rerun the cell")
        ppc_frame = pd.read_csv(ppc_path)
        validate_ppc_results(ppc_frame, k_value, n_value, args.n_test_datasets, args.n_ppc_replicates)
        load_selected_samples(output_dir, k_value, n_value, args)
        frames.append(frame)
        rmse_frames.append(rmse_frame)
        ppc_frames.append(ppc_frame)
        metadata_rows.append(metadata)
    if missing:
        raise FileNotFoundError(
            "Cannot aggregate; missing diagnostic or RMSE SBC cell files: "
            + ", ".join(missing)
            + ". Rerun these cells; old marginal summaries cannot recover RMSE "
            "ranks because joint posterior draws were discarded."
        )

    results = pd.concat(frames, ignore_index=True)
    rmse_results = pd.concat(rmse_frames, ignore_index=True)
    ppc_results = pd.concat(ppc_frames, ignore_index=True)
    results["n_posterior_draws"] = args.n_posterior_draws
    metrics = compute_metrics(results)
    log_gamma_results = compute_log_gamma_metrics(results, args.n_posterior_draws, rmse_results)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    write_csv_atomic(results, output_dir / "diagnostic_results.csv")
    write_csv_atomic(rmse_results, output_dir / "predictive_rmse_sbc_results.csv")
    write_csv_atomic(ppc_results, output_dir / "posterior_predictive_results.csv")
    write_csv_atomic(metrics, output_dir / "diagnostic_metrics.csv")
    write_csv_atomic(log_gamma_results, output_dir / "calibration_log_gamma.csv")
    write_csv_atomic(pd.DataFrame(metadata_rows), output_dir / "cell_runtimes.csv")
    save_all_figures(results, metrics, args, figures_dir, rmse_results, log_gamma_results)
    save_inspection_figures(args, output_dir)

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
        "rmse_sbc": RMSE_SBC_DEFINITION,
        "ppc": PPC_DEFINITION,
        "inspection_dataset": args.inspection_dataset,
        "n_ppc_replicates": args.n_ppc_replicates,
        "seed": args.seed,
        "training_seed": args.training_seed,
        "ecdf_reference_probability": ECDF_REFERENCE_PROBABILITY,
        "ecdf_band_simulations": ECDF_BAND_SIMULATIONS,
        "ecdf_band_seed": ECDF_BAND_SEED,
        "ecdf_band_max_points": ECDF_BAND_MAX_POINTS,
        "log_gamma": LOG_GAMMA_SETTINGS,
        "models_saved": False,
        "posterior_draws_saved": "selected dataset only",
        "training": training_settings(args),
    }
    write_json_atomic(config, output_dir / "config.json")
    (output_dir / "COMPLETE").write_text("complete\n")

    if not args.keep_cell_files:
        shutil.rmtree(cells_dir)
    print(f"Saved STEP 06 diagnostic tables and figures -> {output_dir}")


def replot(args: argparse.Namespace, output_dir: Path) -> None:
    """Regenerate figures from consolidated results without retraining."""
    results_path = output_dir / "diagnostic_results.csv"
    config_path = output_dir / "config.json"
    if not results_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "Replotting requires diagnostic_results.csv and config.json in "
            f"{output_dir}"
        )

    with open(config_path) as handle:
        config = json.load(handle)
    if "log_gamma" in config and config["log_gamma"] != LOG_GAMMA_SETTINGS:
        raise ValueError("Retained log-gamma settings differ from the current calculation")
    args.k_values = tuple(int(value) for value in config["K_values"])
    args.n_values = tuple(int(value) for value in config["N_values"])
    args.n_test_datasets = int(config["M"])
    args.n_posterior_draws = int(config["L"])
    if "ppc" in config:
        if config["ppc"] != PPC_DEFINITION:
            raise ValueError("Retained PPC definition differs from the current check")
        args.inspection_dataset = int(config["inspection_dataset"])
        args.n_ppc_replicates = int(config["n_ppc_replicates"])
        ppc_results = pd.read_csv(output_dir / "posterior_predictive_results.csv")
        if len(ppc_results) != 4 * len(args.k_values) * len(args.n_values) * args.n_test_datasets:
            raise ValueError("Retained PPC table has an unexpected number of rows")
        for k_value, n_value in cell_pairs(args.k_values, args.n_values):
            validate_ppc_results(
                ppc_results.loc[(ppc_results["K"] == k_value) & (ppc_results["N"] == n_value)],
                k_value, n_value, args.n_test_datasets, args.n_ppc_replicates,
            )
    print(
        f"Replotting STEP 06: H=1; M={args.n_test_datasets:,}; "
        f"L={args.n_posterior_draws:,}; "
        f"cells={len(args.k_values) * len(args.n_values)}"
    )

    results = pd.read_csv(results_path)
    rmse_results = None
    rmse_path = output_dir / "predictive_rmse_sbc_results.csv"
    if "rmse_sbc" in config:
        if config["rmse_sbc"] != RMSE_SBC_DEFINITION:
            raise ValueError("Retained RMSE SBC definition differs from the current quantity")
        rmse_results = pd.read_csv(rmse_path)
        if len(rmse_results) != len(args.k_values) * len(args.n_values) * args.n_test_datasets:
            raise ValueError("Retained RMSE SBC table has an unexpected number of rows")
        for k_value, n_value in cell_pairs(args.k_values, args.n_values):
            validate_rmse_results(
                rmse_results.loc[(rmse_results["K"] == k_value) & (rmse_results["N"] == n_value)],
                k_value, n_value, args.n_test_datasets, args.n_posterior_draws,
            )
    else:
        print("Legacy marginal-only results: replotting existing diagnostics without RMSE SBC.")
    metrics = compute_metrics(results)
    log_gamma_results = compute_log_gamma_metrics(results, args.n_posterior_draws, rmse_results)
    write_csv_atomic(metrics, output_dir / "diagnostic_metrics.csv")
    write_csv_atomic(log_gamma_results, output_dir / "calibration_log_gamma.csv")
    save_all_figures(results, metrics, args, output_dir / "figures", rmse_results, log_gamma_results)
    if "ppc" in config:
        save_inspection_figures(args, output_dir)
    else:
        print("Legacy results: posterior plots and PPCs require a new run with retained joint draws.")

    config.update(
        {
            "ecdf_reference_probability": ECDF_REFERENCE_PROBABILITY,
            "ecdf_band_simulations": ECDF_BAND_SIMULATIONS,
            "ecdf_band_seed": ECDF_BAND_SEED,
            "ecdf_band_max_points": ECDF_BAND_MAX_POINTS,
            "log_gamma": LOG_GAMMA_SETTINGS,
        }
    )
    write_json_atomic(config, config_path)
    print(f"Regenerated STEP 06 diagnostic figures -> {output_dir / 'figures'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SBC, recovery, contraction, posterior corner plots, and PPCs for ACE NPEs"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cell-index", type=int, help="One-based K x N array index")
    mode.add_argument("--aggregate", action="store_true")
    mode.add_argument("--replot", action="store_true", help="Replot retained results")
    mode.add_argument("--run-all", action="store_true", help="Local sequential run")
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--n-values", type=int, nargs="+", default=list(DEFAULT_N_VALUES))
    parser.add_argument("--n-test-datasets", type=int, default=1000, metavar="M")
    parser.add_argument("--n-posterior-draws", type=int, default=2000, metavar="L")
    parser.add_argument("--inspection-dataset", type=int, default=1,
                        help="One-based test dataset for corner/PPC plots, shared across K within N")
    parser.add_argument("--n-ppc-replicates", type=int, default=500,
                        help="Noisy PPC replicates per dataset (must be <= L); default: 500")
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
    if not args.replot:
        if not 1 <= args.inspection_dataset <= args.n_test_datasets:
            parser.error("--inspection-dataset must lie between 1 and M")
        if not 2 <= args.n_ppc_replicates <= args.n_posterior_draws:
            parser.error("--n-ppc-replicates must lie between 2 and L")
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
    if args.replot:
        replot(args, output_dir)
        return

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
