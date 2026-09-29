"""STEP 03b -- NPE training-set-size study across fixed twin-pair sample sizes.

For each cell in a K x N grid, this script:

1. generates a Dirichlet ACE simulation corpus in memory;
2. trains one fixed-N NPE (H=1);
3. evaluates the NPE on M shared test datasets using L posterior draws; and
4. saves only compact per-test-dataset summaries.

Training data, posterior draws, and fitted NPE objects are never written to
disk.  The aggregate mode combines all cells, makes the requested distribution
figures, and removes the temporary per-cell files by default.

Notation used throughout the project:

    N  twin-pair sample size
    K  training set size
    h  trained-model index, h=1,...,H (H=1 here)
    m  test-dataset index, m=1,...,M
    l  posterior-draw index, l=1,...,L
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
import torch.nn as nn
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_model import ACE_PARAM_NAMES, COV_FEATURE_NAMES, RESULTS_DIR, resolve
from prior_compare.prior_compare_utils import (
    ACEPosterior,
    SimplexALRPrior,
    ace_to_latent,
)


DEFAULT_K_VALUES = (
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
    300_000,
    500_000,
)
DEFAULT_N_VALUES = (50, 100, 500, 1_000, 2_000, 5_000, 20_000)
METRICS = (
    "error",
    "absolute_error",
    "posterior_kurtosis",
    "posterior_skewness",
    "posterior_sd",
)
METRIC_LABELS = {
    "error": "Estimation error (posterior mean - truth)",
    "absolute_error": "Absolute estimation error",
    "posterior_kurtosis": "Posterior kurtosis",
    "posterior_skewness": "Posterior skewness",
    "posterior_sd": "Posterior SD",
}
METRIC_FILES = {
    "error": "estimation_error_distributions.png",
    "absolute_error": "absolute_error_distributions.png",
    "posterior_kurtosis": "posterior_kurtosis_distributions.png",
    "posterior_skewness": "posterior_skewness_distributions.png",
    "posterior_sd": "posterior_sd_distributions.png",
}


def seed_from(base_seed: int, *keys: int) -> int:
    """Create a reproducible 32-bit seed from a base seed and integer keys."""
    state = np.random.SeedSequence([base_seed, *keys]).generate_state(1)
    return int(state[0])


def _wishart_compound_symmetric_summaries(
    correlations: np.ndarray,
    n_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw exact Gaussian sample-covariance summaries without raw twin data.

    For centered bivariate Gaussian observations, (N-1)S follows a Wishart
    distribution.  A vectorized Bartlett decomposition therefore produces the
    same distribution as ``np.cov`` on N explicitly simulated twin pairs, but
    its cost does not grow with N.
    """
    correlations = np.asarray(correlations, dtype=np.float64)
    if correlations.ndim != 1:
        raise ValueError("correlations must be one-dimensional")
    if n_pairs < 2:
        raise ValueError("n_pairs must be at least 2")
    if np.any(np.abs(correlations) >= 1.0):
        raise ValueError("correlations must lie strictly between -1 and 1")

    degrees_freedom = n_pairs - 1
    # Separate streams keep every random vector prefix-stable when K changes.
    a11_rng = np.random.default_rng(seed_from(seed, 1))
    a21_rng = np.random.default_rng(seed_from(seed, 2))
    a22_rng = np.random.default_rng(seed_from(seed, 3))
    a11 = np.sqrt(a11_rng.chisquare(degrees_freedom, size=len(correlations)))
    a21 = a21_rng.standard_normal(size=len(correlations))
    a22 = np.sqrt(
        a22_rng.chisquare(degrees_freedom - 1, size=len(correlations))
    )

    # Cholesky([[1, rho], [rho, 1]]) @ Bartlett factor.
    l22 = np.sqrt(np.maximum(1.0 - correlations**2, 0.0))
    b11 = a11
    b21 = correlations * a11 + l22 * a21
    b22 = l22 * a22

    s11 = b11**2 / degrees_freedom
    s12 = b11 * b21 / degrees_freedom
    s22 = (b21**2 + b22**2) / degrees_freedom
    return s11, s12, s22


def simulate_fixed_n(
    n_simulations: int,
    n_pairs: int,
    seed: int,
    return_covariances: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return four covariance features and Dirichlet(1,1,1) ACE truths.

    The random streams are separated so calls with the same ``seed`` and
    different sizes share prefixes.  Thus the K=10k corpus is nested inside
    the K=20k corpus, and so on, for a fixed N.
    """
    if n_simulations < 2:
        raise ValueError("n_simulations must be at least 2")

    theta_rng = np.random.default_rng(seed_from(seed, 1))
    ace = theta_rng.dirichlet(np.ones(3), size=n_simulations)

    mz_correlation = ace[:, 0] + ace[:, 1]
    dz_correlation = 0.5 * ace[:, 0] + ace[:, 1]
    mz_var1, mz_cov, mz_var2 = _wishart_compound_symmetric_summaries(
        mz_correlation, n_pairs, seed_from(seed, 2)
    )
    dz_var1, dz_cov, dz_var2 = _wishart_compound_symmetric_summaries(
        dz_correlation, n_pairs, seed_from(seed, 3)
    )
    mz_var = 0.5 * (mz_var1 + mz_var2)
    dz_var = 0.5 * (dz_var1 + dz_var2)
    features = np.column_stack((mz_var, mz_cov, dz_var, dz_cov)).astype(
        np.float32
    )
    ace = ace.astype(np.float32)
    if not return_covariances:
        return features, ace
    covariance_data = pd.DataFrame(
        {
            "mz_var1": mz_var1,
            "mz_cov": mz_cov,
            "mz_var2": mz_var2,
            "dz_var1": dz_var1,
            "dz_cov": dz_cov,
            "dz_var2": dz_var2,
            "mz_var": mz_var,
            "dz_var": dz_var,
        }
    )
    return features, ace, covariance_data


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def train_transient_posterior(
    features: np.ndarray,
    ace: np.ndarray,
    args: argparse.Namespace,
) -> tuple[ACEPosterior, StandardScaler]:
    """Train one in-memory Dirichlet NPE; do not serialize the fitted model."""
    n_rows = len(features)
    n_scaler_train = max(2, int((1.0 - args.validation_fraction) * n_rows))
    scaler = StandardScaler().fit(features[:n_scaler_train])
    scaled = scaler.transform(features).astype(np.float32)
    theta = ace_to_latent(ace, "dirichlet")

    prior = SimplexALRPrior(
        "dirichlet", (1.0, 1.0, 1.0), device=args.device_resolved
    )
    torch.manual_seed(args.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.training_seed)
    density_builder = posterior_nn(
        model=args.flow_type,
        embedding_net=nn.Identity(),
        hidden_features=args.flow_hidden,
        num_transforms=args.flow_transforms,
        z_score_theta="independent",
        z_score_x="independent",
    )
    inference = SNPE(
        prior=prior,
        density_estimator=density_builder,
        device=str(args.device_resolved),
    )
    inference.append_simulations(
        theta=torch.as_tensor(theta, dtype=torch.float32),
        x=torch.as_tensor(scaled, dtype=torch.float32),
    )
    estimator = inference.train(
        training_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_num_epochs=args.epochs,
        stop_after_epochs=args.stop_after_epochs,
        validation_fraction=args.validation_fraction,
        show_train_summary=True,
    )
    latent_posterior = inference.build_posterior(estimator)
    latent_posterior.to("cpu")
    return ACEPosterior(latent_posterior, "dirichlet", 1.0), scaler


def evaluate_cell(
    posterior: ACEPosterior,
    scaler: StandardScaler,
    test_features: np.ndarray,
    test_ace: np.ndarray,
    k_value: int,
    n_pairs: int,
    n_draws: int,
    posterior_seed: int,
) -> pd.DataFrame:
    """Create one compact row per test dataset and ACE parameter."""
    torch.manual_seed(posterior_seed)
    scaled = scaler.transform(test_features).astype(np.float32)
    rows: list[dict[str, float | int | str]] = []
    for test_index, (feature_row, truth_row) in enumerate(
        zip(scaled, test_ace), start=1
    ):
        x = torch.as_tensor(feature_row, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            draws = posterior.sample(
                (n_draws,), x=x, show_progress_bars=False
            ).cpu().numpy()
        for parameter_index, parameter in enumerate(ACE_PARAM_NAMES):
            values = draws[:, parameter_index]
            truth = float(truth_row[parameter_index])
            posterior_mean = float(values.mean())
            error = posterior_mean - truth
            rows.append(
                {
                    "K": k_value,
                    "N": n_pairs,
                    "model_index_h": 1,
                    "test_dataset_m": test_index,
                    "parameter": parameter,
                    "truth": truth,
                    "posterior_mean": posterior_mean,
                    "error": error,
                    "absolute_error": abs(error),
                    "posterior_sd": float(values.std(ddof=1)),
                    "posterior_skewness": float(skew(values, bias=False)),
                    "posterior_kurtosis": float(
                        kurtosis(values, fisher=False, bias=False)
                    ),
                    "ci_2_5": float(np.percentile(values, 2.5)),
                    "ci_97_5": float(np.percentile(values, 97.5)),
                    "true_rank": int(np.sum(values < truth)),
                    "true_rank_percentile": float(np.mean(values < truth)),
                }
            )
    return pd.DataFrame(rows)


def cell_pairs(k_values: tuple[int, ...], n_values: tuple[int, ...]):
    """Return cells with K varying fastest inside each N."""
    return tuple((k_value, n_value) for n_value in n_values for k_value in k_values)


def cell_stem(k_value: int, n_value: int) -> str:
    return f"K{k_value}_N{n_value}"


def write_json_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run_cell(
    k_value: int,
    n_value: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    stem = cell_stem(k_value, n_value)
    results_path = cells_dir / f"{stem}.csv"
    metadata_path = cells_dir / f"{stem}.json"
    if results_path.exists() and metadata_path.exists() and not args.overwrite:
        print(f"Skipping completed cell K={k_value:,}, N={n_value:,}")
        return

    # A newly launched cell means this output directory is no longer complete.
    (output_dir / "COMPLETE").unlink(missing_ok=True)

    started = time.perf_counter()
    print(f"Training transient NPE: K={k_value:,}, N={n_value:,}, H=1")
    training_seed = seed_from(args.seed, n_value, 10)
    features, ace = simulate_fixed_n(k_value, n_value, training_seed)
    posterior, scaler = train_transient_posterior(features, ace, args)

    # Test data depend on N, but not K, so all K values use identical cases.
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
        "validation_fraction": args.validation_fraction,
        "approximate_optimizer_training_rows": int(
            round((1.0 - args.validation_fraction) * k_value)
        ),
        "elapsed_seconds": elapsed_seconds,
        "device": str(args.device_resolved),
        "models_saved": False,
        "training_data_saved": False,
        "posterior_draws_saved": False,
    }
    write_csv_atomic(results, results_path)
    write_json_atomic(metadata, metadata_path)
    print(
        f"Completed K={k_value:,}, N={n_value:,} in "
        f"{elapsed_seconds / 60.0:.1f} minutes -> {results_path}"
    )

    del posterior, scaler, features, ace, test_features, test_ace, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize_results(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for (k_value, n_value, parameter), group in results.groupby(
        ["K", "N", "parameter"], sort=True
    ):
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            summary_rows.append(
                {
                    "K": int(k_value),
                    "N": int(n_value),
                    "model_index_h": 1,
                    "parameter": parameter,
                    "metric": metric,
                    "n_test_datasets": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            )
    long_summary = pd.DataFrame(summary_rows)

    cell_rows = []
    for (k_value, n_value, parameter), group in results.groupby(
        ["K", "N", "parameter"], sort=True
    ):
        covered = (group["truth"] >= group["ci_2_5"]) & (
            group["truth"] <= group["ci_97_5"]
        )
        cell_rows.append(
            {
                "K": int(k_value),
                "N": int(n_value),
                "model_index_h": 1,
                "parameter": parameter,
                "M": len(group),
                "bias": float(group["error"].mean()),
                "mae": float(group["absolute_error"].mean()),
                "rmse": float(np.sqrt(np.mean(group["error"] ** 2))),
                "mean_posterior_sd": float(group["posterior_sd"].mean()),
                "coverage_95": float(covered.mean()),
            }
        )
    return long_summary, pd.DataFrame(cell_rows)


def compact_number(value: int) -> str:
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}k"
    return str(value)


def plot_metric_distributions(
    results: pd.DataFrame,
    metric: str,
    k_values: tuple[int, ...],
    n_values: tuple[int, ...],
    n_test_datasets: int,
    n_draws: int,
    path: Path,
) -> None:
    """Plot distributions across M test datasets for every K, N, and parameter."""
    fig, axes = plt.subplots(
        len(ACE_PARAM_NAMES),
        len(n_values),
        figsize=(3.4 * len(n_values), 10.5),
        squeeze=False,
    )
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(k_values)))
    positions = np.arange(len(k_values))

    for row_index, parameter in enumerate(ACE_PARAM_NAMES):
        for column_index, n_value in enumerate(n_values):
            axis = axes[row_index, column_index]
            distributions = []
            for k_value in k_values:
                values = results.loc[
                    (results["K"] == k_value)
                    & (results["N"] == n_value)
                    & (results["parameter"] == parameter),
                    metric,
                ].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                distributions.append(values)
            boxes = axis.boxplot(
                distributions,
                positions=positions,
                widths=0.62,
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "black", "linewidth": 1.1},
                flierprops={
                    "marker": ".",
                    "markersize": 1.5,
                    "markerfacecolor": "0.35",
                    "markeredgecolor": "0.35",
                    "alpha": 0.18,
                },
            )
            for box, color in zip(boxes["boxes"], colors):
                box.set_facecolor(color)
                box.set_edgecolor(color)
                box.set_alpha(0.55)
            if metric in {"error", "posterior_skewness"}:
                axis.axhline(0.0, color="firebrick", linestyle="--", linewidth=0.9)
            elif metric == "posterior_kurtosis":
                axis.axhline(3.0, color="firebrick", linestyle="--", linewidth=0.9)
                if all(np.all(values > 0) for values in distributions if len(values)):
                    axis.set_yscale("log")
            else:
                axis.set_ylim(bottom=0.0)
            if row_index == 0:
                axis.set_title(f"N={n_value:,}")
            if column_index == 0:
                axis.set_ylabel(f"{parameter}\n{METRIC_LABELS[metric]}")
            if row_index == len(ACE_PARAM_NAMES) - 1:
                axis.set_xlabel("Training set size K")
            axis.set_xticks(positions, [compact_number(value) for value in k_values])
            axis.tick_params(axis="x", labelrotation=45, labelsize=8)
            axis.ticklabel_format(
                axis="y", style="sci", scilimits=(-3, 3), useMathText=True
            ) if metric != "posterior_kurtosis" else None
            axis.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"{METRIC_LABELS[metric]} across training set sizes and sample sizes\n"
        f"H=1 model per (K,N); M={n_test_datasets:,} test datasets; "
        f"L={n_draws:,} posterior draws",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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
        frames.append(frame)
        with open(metadata_path) as handle:
            metadata_rows.append(json.load(handle))
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} of {len(pairs)} cells: {preview}{suffix}"
        )

    results = pd.concat(frames, ignore_index=True)
    results.sort_values(
        ["N", "K", "test_dataset_m", "parameter"], inplace=True
    )
    summary, cell_summary = summarize_results(results)
    write_csv_atomic(results, output_dir / "training_budget_test_summaries.csv")
    write_csv_atomic(summary, output_dir / "training_budget_distribution_summary.csv")
    write_csv_atomic(cell_summary, output_dir / "training_budget_cell_summary.csv")
    write_csv_atomic(pd.DataFrame(metadata_rows), output_dir / "cell_runtimes.csv")

    for metric in METRICS:
        print(f"Plotting {METRIC_LABELS[metric]} ...")
        plot_metric_distributions(
            results,
            metric,
            args.k_values,
            args.n_values,
            args.n_test_datasets,
            args.n_posterior_draws,
            output_dir / METRIC_FILES[metric],
        )

    config = {
        "experiment": "fixed_N_training_set_size_grid",
        "notation": {
            "N": "twin-pair sample size",
            "K": "training set size before internal validation split",
            "H": "number of independently trained models per (K,N)",
            "M": "number of test datasets per (K,N)",
            "L": "posterior draws per test dataset",
        },
        "K_values": list(args.k_values),
        "N_values": list(args.n_values),
        "H": 1,
        "M": args.n_test_datasets,
        "L": args.n_posterior_draws,
        "prior": "Dirichlet(1,1,1), with A+C+E=1",
        "simulation": (
            "Exact Wishart sample-covariance draws; no raw twin pairs stored"
        ),
        "shared_tests": "Identical M test datasets across K for each N",
        "nested_training_data": "Training simulations share prefixes across K for each N",
        "models_saved": False,
        "training_data_saved": False,
        "posterior_draws_saved": False,
        "final_detailed_output": "training_budget_test_summaries.csv",
        "validation_fraction": args.validation_fraction,
        "flow_type": args.flow_type,
        "flow_hidden": args.flow_hidden,
        "flow_transforms": args.flow_transforms,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "stop_after_epochs": args.stop_after_epochs,
        "seed": args.seed,
        "training_seed": args.training_seed,
    }
    write_json_atomic(config, output_dir / "config.json")
    (output_dir / "COMPLETE").write_text("complete\n")

    if not args.keep_cell_files:
        shutil.rmtree(cells_dir)
        print(f"Removed temporary cell files: {cells_dir}")
    print(f"Saved complete training-set-size study -> {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study NPE training set size K across fixed sample sizes N"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--cell-index",
        type=int,
        help="One-based K x N cell index, suitable for a SLURM array",
    )
    mode.add_argument("--aggregate", action="store_true")
    mode.add_argument(
        "--run-all",
        action="store_true",
        help="Run every cell sequentially and then aggregate (mainly for local tests)",
    )
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--n-values", type=int, nargs="+", default=list(DEFAULT_N_VALUES))
    parser.add_argument("--n-test-datasets", type=int, default=500, metavar="M")
    parser.add_argument("--n-posterior-draws", type=int, default=2000, metavar="L")
    parser.add_argument("--output-dir", default="training_budget_grid")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--stop-after-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--flow-type", choices=("nsf", "maf", "maf_rqs", "mdn"), default="nsf")
    parser.add_argument("--flow-hidden", type=int, default=64)
    parser.add_argument("--flow-transforms", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-cell-files",
        action="store_true",
        help="Keep temporary per-cell CSV/JSON files after successful aggregation",
    )
    args = parser.parse_args()

    args.k_values = tuple(dict.fromkeys(args.k_values))
    args.n_values = tuple(dict.fromkeys(args.n_values))
    if any(value < 100 for value in args.k_values):
        parser.error("every K value must be at least 100")
    if any(value < 2 for value in args.n_values):
        parser.error("every N value must be at least 2")
    if args.n_test_datasets < 2:
        parser.error("M (--n-test-datasets) must be at least 2")
    if args.n_posterior_draws < 20:
        parser.error("L (--n-posterior-draws) must be at least 20")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must lie between 0 and 0.5")
    if args.epochs < 1 or args.stop_after_epochs < 1 or args.batch_size < 1:
        parser.error("epochs, stop-after-epochs, and batch-size must be positive")
    if args.cell_index is not None:
        total_cells = len(args.k_values) * len(args.n_values)
        if not 1 <= args.cell_index <= total_cells:
            parser.error(f"--cell-index must lie between 1 and {total_cells}")
    args.device_resolved = resolve_device(args.device)
    return args


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir, RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = cell_pairs(args.k_values, args.n_values)
    print(
        f"Notation: H=1; M={args.n_test_datasets:,} test datasets; "
        f"L={args.n_posterior_draws:,} posterior draws"
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
