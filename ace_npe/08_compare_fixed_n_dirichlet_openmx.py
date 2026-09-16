"""STEP 08 -- paired OpenMx versus fixed-N Dirichlet NPE comparison.

For each Dirichlet ACE condition and sample size, this script simulates one MZ
and one DZ covariance matrix. OpenMx and the matching fixed-N NPE receive those
exact same covariance matrices. Metrics for both methods use the same subset of
rows on which OpenMx converged, so differences are genuinely paired.

The script calls ``08_fit_openmx_paired_dirichlet.R`` as its OpenMx backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binom, kurtosis, skew

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_model import ACE_PARAM_NAMES, COV_FEATURE_NAMES, MODELS_DIR, RESULTS_DIR, load_posterior, resolve
from prior_compare.prior_compare_utils import model_dir_for


DEFAULT_N_PAIRS = (50, 100, 500, 1000, 2000, 5000, 20000)
METHOD_COLORS = {"OpenMx": "tab:blue", "NPE": "tab:orange"}


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_models(models_dir: Path, n_values: tuple[int, ...]) -> None:
    missing = []
    for n_pairs in n_values:
        model_dir = model_dir_for(models_dir, "dirichlet", n_pairs)
        required = ("config.json", "posterior.pkl", "feature_scaler.pkl")
        if any(not (model_dir / filename).exists() for filename in required):
            missing.append(n_pairs)
    if missing:
        commands = "\n".join(
            "  python prior_compare/01_generate_prior_data.py --schemes dirichlet "
            f"--n_pairs {n} --skip_manifest\n"
            "  python prior_compare/03_train_prior_models.py --schemes dirichlet "
            f"--n_pairs {n}"
            for n in missing
        )
        raise FileNotFoundError(
            f"Missing fixed-N Dirichlet NPE models for N={missing}. "
            f"Generate and train them first:\n{commands}"
        )


def sample_covariance(
    covariance: np.ndarray,
    n_pairs: int,
    seed_parts: tuple[int, ...],
) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
    pairs = rng.multivariate_normal(
        mean=np.zeros(2), cov=covariance, size=n_pairs, check_valid="raise"
    )
    return np.cov(pairs, rowvar=False, ddof=1)


def generate_paired_data(
    n_conditions: int,
    n_values: tuple[int, ...],
    alpha: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """Create one shared Dirichlet test dataset for both estimators."""
    theta_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    truths = theta_rng.dirichlet(alpha, size=n_conditions)
    rows = []
    for condition_index, (a, c, e) in enumerate(truths, start=1):
        total = a + c + e
        mz_population = np.array([[total, a + c], [a + c, total]])
        dz_population = np.array([[total, 0.5 * a + c], [0.5 * a + c, total]])
        for n_pairs in n_values:
            mz = sample_covariance(
                mz_population, n_pairs, (seed, condition_index, n_pairs, 1)
            )
            dz = sample_covariance(
                dz_population, n_pairs, (seed, condition_index, n_pairs, 2)
            )
            rows.append(
                {
                    "condition_id": condition_index,
                    "N_pairs": n_pairs,
                    "A_true": a,
                    "C_true": c,
                    "E_true": e,
                    "mz_var1": mz[0, 0],
                    "mz_cov": mz[0, 1],
                    "mz_var2": mz[1, 1],
                    "dz_var1": dz[0, 0],
                    "dz_cov": dz[0, 1],
                    "dz_var2": dz[1, 1],
                    # These are the exact sufficient reductions used by the NPE.
                    "mz_var": np.diag(mz).mean(),
                    "dz_var": np.diag(dz).mean(),
                }
            )
    return pd.DataFrame(rows)


def check_paired_data(
    paired: pd.DataFrame,
    n_conditions: int,
    n_values: tuple[int, ...],
) -> None:
    required = {
        "condition_id", "N_pairs", "A_true", "C_true", "E_true",
        "mz_var1", "mz_cov", "mz_var2", "dz_var1", "dz_cov", "dz_var2",
        "mz_var", "dz_var",
    }
    missing = sorted(required.difference(paired.columns))
    if missing:
        raise ValueError(f"Paired data are missing columns: {missing}")
    if len(paired) != n_conditions * len(n_values):
        raise ValueError(
            f"Expected {n_conditions * len(n_values)} paired rows, found {len(paired)}"
        )
    if set(paired["N_pairs"].astype(int)) != set(n_values):
        raise ValueError("Reused paired data do not contain the requested N values")
    counts = paired.groupby("N_pairs")["condition_id"].nunique()
    if not (counts == n_conditions).all():
        raise ValueError("Each N must contain exactly n_conditions unique conditions")
    if not np.allclose(
        paired[["A_true", "C_true", "E_true"]].sum(axis=1), 1.0, atol=1e-10
    ):
        raise ValueError("Dirichlet truths do not sum to one")


def run_openmx(
    rscript: str,
    backend: Path,
    paired_path: Path,
    output_path: Path,
) -> None:
    if not backend.exists():
        raise FileNotFoundError(f"OpenMx backend not found: {backend}")
    command = [rscript, str(backend), str(paired_path), str(output_path)]
    print("Running OpenMx on the paired covariance matrices ...")
    subprocess.run(command, check=True)


def evaluate_npe(
    paired: pd.DataFrame,
    models_dir: Path,
    n_values: tuple[int, ...],
    n_draws: int,
    seed: int,
) -> pd.DataFrame:
    torch.manual_seed(seed)
    frames = []
    for n_pairs in n_values:
        print(f"Evaluating Dirichlet NPE, fixed N={n_pairs} ...")
        subset = paired.loc[paired["N_pairs"] == n_pairs].copy()
        loaded = load_posterior(model_dir_for(models_dir, "dirichlet", n_pairs))
        config = loaded["config"]
        if config.get("simulation_scheme") != "dirichlet":
            raise ValueError(f"N={n_pairs} model is not a Dirichlet model")
        if config.get("fixed_n_pairs") != n_pairs:
            raise ValueError(
                f"N={n_pairs} directory contains a fixed-N={config.get('fixed_n_pairs')} model"
            )
        if config.get("n_is_model_feature") is not False:
            raise ValueError(f"N={n_pairs} model unexpectedly uses N as a feature")
        if list(loaded["feature_cols"]) != list(COV_FEATURE_NAMES):
            raise ValueError(
                f"N={n_pairs} model features are {loaded['feature_cols']}, "
                f"expected {list(COV_FEATURE_NAMES)}"
            )

        features = subset[COV_FEATURE_NAMES].to_numpy(dtype=np.float32)
        scaled = loaded["scaler"].transform(features)
        truths = subset[[f"{p}_true" for p in ACE_PARAM_NAMES]].to_numpy()
        output = subset[["condition_id", "N_pairs"]].reset_index(drop=True)

        summaries = {p: [] for p in ACE_PARAM_NAMES}
        for row_index, row in enumerate(scaled):
            x = torch.as_tensor(row, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                samples = loaded["posterior"].sample(
                    (n_draws,), x=x, show_progress_bars=False
                )
            draws = samples.cpu().numpy()
            truth = truths[row_index]
            for parameter_index, parameter in enumerate(ACE_PARAM_NAMES):
                values = draws[:, parameter_index]
                rank = int(np.sum(values < truth[parameter_index]))
                summaries[parameter].append(
                    {
                        "mean": values.mean(),
                        "sd": values.std(ddof=1),
                        "min": values.min(),
                        "max": values.max(),
                        "median": np.median(values),
                        "ci_lo": np.percentile(values, 2.5),
                        "ci_hi": np.percentile(values, 97.5),
                        "skewness": skew(values, bias=False),
                        "excess_kurtosis": kurtosis(values, fisher=True, bias=False),
                        "true_rank": rank,
                        "true_percentile": rank / n_draws,
                    }
                )

        for parameter in ACE_PARAM_NAMES:
            summary = pd.DataFrame(summaries[parameter])
            for statistic in summary.columns:
                output[f"{parameter}_npe_{statistic}"] = summary[statistic].to_numpy()
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def merge_results(
    paired: pd.DataFrame,
    npe: pd.DataFrame,
    openmx: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["condition_id", "N_pairs"]
    if openmx.duplicated(keys).any() or npe.duplicated(keys).any():
        raise ValueError("Estimator results contain duplicate condition/N keys")
    merged = paired.merge(npe, on=keys, how="left", validate="one_to_one")
    merged = merged.merge(openmx, on=keys, how="left", validate="one_to_one")
    if merged.filter(regex="_npe_mean$").isna().any().any():
        raise ValueError("Some paired rows have no NPE estimates")
    if merged["openmx_converged"].isna().any():
        raise ValueError("Some paired rows have no OpenMx result")
    return merged


def metric_tables(
    merged: pd.DataFrame,
    n_values: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    difference_rows = []
    for n_pairs in n_values:
        at_n = merged.loc[merged["N_pairs"] == n_pairs].copy()
        finite_openmx = np.isfinite(
            at_n[[f"{p}_openmx_est" for p in ACE_PARAM_NAMES]]
        ).all(axis=1)
        eligible = at_n.loc[
            at_n["openmx_converged"].eq(1) & finite_openmx
        ].copy()
        if eligible.empty:
            raise RuntimeError(f"OpenMx had no usable fits at N={n_pairs}")

        for parameter in ACE_PARAM_NAMES:
            truth = eligible[f"{parameter}_true"].to_numpy()
            estimates = {
                "OpenMx": eligible[f"{parameter}_openmx_est"].to_numpy(),
                "NPE": eligible[f"{parameter}_npe_mean"].to_numpy(),
            }
            for method, estimate in estimates.items():
                error = estimate - truth
                bias = float(error.mean())
                bias_se = float(error.std(ddof=1) / np.sqrt(len(error)))
                absolute_error = np.abs(error)
                mae = float(absolute_error.mean())
                mae_se = float(
                    absolute_error.std(ddof=1) / np.sqrt(len(absolute_error))
                )
                if method == "OpenMx":
                    se = eligible[f"{parameter}_openmx_se"].to_numpy()
                    finite_interval = np.isfinite(se)
                    covered = (
                        (truth[finite_interval] >= estimate[finite_interval] - 1.96 * se[finite_interval])
                        & (truth[finite_interval] <= estimate[finite_interval] + 1.96 * se[finite_interval])
                    )
                else:
                    low = eligible[f"{parameter}_npe_ci_lo"].to_numpy()
                    high = eligible[f"{parameter}_npe_ci_hi"].to_numpy()
                    se = eligible[f"{parameter}_npe_sd"].to_numpy()
                    finite_interval = np.isfinite(low) & np.isfinite(high)
                    covered = (
                        (truth[finite_interval] >= low[finite_interval])
                        & (truth[finite_interval] <= high[finite_interval])
                    )
                finite_uncertainty = se[np.isfinite(se)]
                mean_uncertainty = float(finite_uncertainty.mean())
                uncertainty_mean_se = float(
                    finite_uncertainty.std(ddof=1)
                    / np.sqrt(len(finite_uncertainty))
                )
                n_covered = int(covered.sum())
                coverage_ci_lo, coverage_ci_hi = wilson_interval(
                    n_covered, len(covered)
                )
                metric_rows.append(
                    {
                        "method": method,
                        "N_pairs": n_pairs,
                        "parameter": parameter,
                        "n_total": len(at_n),
                        "n_paired": len(eligible),
                        "openmx_convergence_rate": len(eligible) / len(at_n),
                        "bias": bias,
                        "bias_se": bias_se,
                        "bias_ci_lo": bias - 1.96 * bias_se,
                        "bias_ci_hi": bias + 1.96 * bias_se,
                        "mae": mae,
                        "mae_se": mae_se,
                        "mae_ci_lo": max(0.0, mae - 1.96 * mae_se),
                        "mae_ci_hi": mae + 1.96 * mae_se,
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "uncertainty_measure": (
                            "standard error"
                            if method == "OpenMx"
                            else "posterior standard deviation"
                        ),
                        "mean_uncertainty": mean_uncertainty,
                        "uncertainty_mean_se": uncertainty_mean_se,
                        "uncertainty_mean_ci_lo": max(
                            0.0, mean_uncertainty - 1.96 * uncertainty_mean_se
                        ),
                        "uncertainty_mean_ci_hi": (
                            mean_uncertainty + 1.96 * uncertainty_mean_se
                        ),
                        "n_uncertainty": int(len(finite_uncertainty)),
                        "coverage_95": float(covered.mean()) if len(covered) else np.nan,
                        "coverage_95_ci_lo": coverage_ci_lo,
                        "coverage_95_ci_hi": coverage_ci_hi,
                        "n_covered": n_covered,
                        "n_coverage": int(len(covered)),
                    }
                )

            error_difference = estimates["NPE"] - estimates["OpenMx"]
            difference = float(error_difference.mean())
            difference_se = float(
                error_difference.std(ddof=1) / np.sqrt(len(error_difference))
            )
            difference_rows.append(
                {
                    "N_pairs": n_pairs,
                    "parameter": parameter,
                    "n_paired": len(eligible),
                    "mean_npe_minus_openmx_estimate": difference,
                    "difference_se": difference_se,
                    "difference_ci_lo": difference - 1.96 * difference_se,
                    "difference_ci_hi": difference + 1.96 * difference_se,
                    "rmse_between_estimators": float(
                        np.sqrt(np.mean(error_difference**2))
                    ),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(difference_rows)


def bias_norm_table(
    merged: pd.DataFrame,
    n_values: tuple[int, ...],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Calculate ||mean(error_A, error_C, error_E)|| with paired bootstrap CIs."""
    rows = []
    for n_pairs in n_values:
        at_n = merged.loc[merged["N_pairs"] == n_pairs].copy()
        finite_openmx = np.isfinite(
            at_n[[f"{p}_openmx_est" for p in ACE_PARAM_NAMES]]
        ).all(axis=1)
        eligible = at_n.loc[
            at_n["openmx_converged"].eq(1) & finite_openmx
        ].copy()
        if eligible.empty:
            raise RuntimeError(f"OpenMx had no usable fits at N={n_pairs}")

        rng = np.random.default_rng(np.random.SeedSequence([seed, n_pairs, 3]))
        bootstrap_indices = rng.integers(
            0, len(eligible), size=(n_bootstrap, len(eligible))
        )
        truth = eligible[[f"{p}_true" for p in ACE_PARAM_NAMES]].to_numpy()
        for method, suffix in (("OpenMx", "openmx_est"), ("NPE", "npe_mean")):
            estimates = eligible[
                [f"{p}_{suffix}" for p in ACE_PARAM_NAMES]
            ].to_numpy()
            errors = estimates - truth
            bias_vector = errors.mean(axis=0)
            bootstrap_norms = np.linalg.norm(
                errors[bootstrap_indices].mean(axis=1), axis=1
            )
            rows.append(
                {
                    "method": method,
                    "N_pairs": n_pairs,
                    "n_paired": len(eligible),
                    "bias_A": bias_vector[0],
                    "bias_C": bias_vector[1],
                    "bias_E": bias_vector[2],
                    "bias_vector_norm": float(np.linalg.norm(bias_vector)),
                    "bias_vector_norm_ci_lo": float(
                        np.percentile(bootstrap_norms, 2.5)
                    ),
                    "bias_vector_norm_ci_hi": float(
                        np.percentile(bootstrap_norms, 97.5)
                    ),
                    "n_bootstrap": n_bootstrap,
                }
            )
    return pd.DataFrame(rows)


def configure_n_axis(
    axis,
    n_values: tuple[int, ...],
    xlabel: str = "Twin-pair sample size N (per zygosity group)",
) -> None:
    axis.set_xscale("log")
    axis.set_xticks(n_values, [str(n) for n in n_values])
    axis.set_xlabel(xlabel)
    axis.grid(alpha=0.25)


def save_bias_plot(
    metrics: pd.DataFrame,
    n_values: tuple[int, ...],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), squeeze=False)
    last_n = n_values[-1]
    test_set_size = int(metrics["n_total"].max())
    for col, parameter in enumerate(ACE_PARAM_NAMES):
        axis = axes[0, col]
        at_parameter = metrics.loc[metrics["parameter"] == parameter]
        limit = 1.08 * np.abs(
            at_parameter[["bias_ci_lo", "bias_ci_hi"]].to_numpy()
        ).max()
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
        for method in ("OpenMx", "NPE"):
            subset = (
                at_parameter.loc[at_parameter["method"] == method]
                .set_index("N_pairs")
                .reindex(n_values)
            )
            color = METHOD_COLORS[method]
            axis.fill_between(
                n_values,
                subset["bias_ci_lo"],
                subset["bias_ci_hi"],
                color=color,
                alpha=0.14,
            )
            axis.plot(
                n_values,
                subset["bias"],
                marker="o",
                color=color,
                label=method,
            )
            last_value = float(subset["bias"].iloc[-1])
            axis.annotate(
                f"{last_value:.5f}",
                xy=(last_n, last_value),
                xytext=(0, 10 if method == "OpenMx" else -16),
                textcoords="offset points",
                ha="right",
                color=color,
                fontsize=9,
            )
        axis.set_ylim(-limit, limit)
        axis.set_title(parameter)
        axis.set_ylabel("Bias (estimate - truth)")
        configure_n_axis(axis, n_values, xlabel="Twin-pair sample size N")

    axes[0, 0].legend()
    fig.suptitle(
        "Paired Dirichlet bias: OpenMx versus fixed-N NPE "
        f"(test set size = {test_set_size:,} per N)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_performance_plot(
    metrics: pd.DataFrame,
    n_values: tuple[int, ...],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), squeeze=False)
    last_n = n_values[-1]
    test_set_size = int(metrics["n_total"].max())
    for col, parameter in enumerate(ACE_PARAM_NAMES):
        axis = axes[0, col]
        for method in ("OpenMx", "NPE"):
            subset = (
                metrics.loc[
                    (metrics["method"] == method)
                    & (metrics["parameter"] == parameter)
                ]
                .set_index("N_pairs")
                .reindex(n_values)
            )
            values = subset["rmse"].to_numpy()
            color = METHOD_COLORS[method]
            axis.plot(
                n_values,
                values,
                marker="o",
                color=color,
                label=method,
            )
            last_value = float(values[-1])
            axis.annotate(
                f"{last_value:.5f}",
                xy=(last_n, last_value),
                xytext=(0, 10 if method == "OpenMx" else -16),
                textcoords="offset points",
                ha="right",
                color=color,
                fontsize=9,
            )
        axis.set_title(parameter)
        axis.set_ylabel("RMSE")
        configure_n_axis(axis, n_values, xlabel="Twin-pair sample size N")
    axes[0, 0].legend()
    fig.suptitle(
        "Paired Dirichlet RMSE: OpenMx versus fixed-N NPE "
        f"(test set size = {test_set_size:,} per N)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_mae_plot(metrics: pd.DataFrame, n_values: tuple[int, ...], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), squeeze=False)
    last_n = n_values[-1]
    test_set_size = int(metrics["n_total"].max())
    for col, parameter in enumerate(ACE_PARAM_NAMES):
        axis = axes[0, col]
        at_parameter = metrics.loc[metrics["parameter"] == parameter]
        for method in ("OpenMx", "NPE"):
            subset = (
                at_parameter.loc[at_parameter["method"] == method]
                .set_index("N_pairs")
                .reindex(n_values)
            )
            color = METHOD_COLORS[method]
            axis.fill_between(
                n_values,
                subset["mae_ci_lo"],
                subset["mae_ci_hi"],
                color=color,
                alpha=0.14,
            )
            axis.plot(
                n_values,
                subset["mae"],
                marker="o",
                color=color,
                label=method,
            )
            last_value = float(subset["mae"].iloc[-1])
            axis.annotate(
                f"{last_value:.5f}",
                xy=(last_n, last_value),
                xytext=(0, 10 if method == "OpenMx" else -16),
                textcoords="offset points",
                ha="right",
                color=color,
                fontsize=9,
            )
        axis.set_ylim(bottom=0.0)
        axis.set_title(parameter)
        axis.set_ylabel("Mean absolute error")
        configure_n_axis(axis, n_values, xlabel="Twin-pair sample size N")
    axes[0, 0].legend()
    fig.suptitle(
        "Paired Dirichlet mean absolute error: OpenMx versus fixed-N NPE "
        f"(test set size = {test_set_size:,} per N)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_mean_uncertainty_plot(
    metrics: pd.DataFrame,
    n_values: tuple[int, ...],
    path: Path,
) -> None:
    """Compare mean OpenMx SE with mean NPE posterior SD."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), squeeze=False)
    last_n = n_values[-1]
    test_set_size = int(metrics["n_total"].max())
    for col, parameter in enumerate(ACE_PARAM_NAMES):
        axis = axes[0, col]
        at_parameter = metrics.loc[metrics["parameter"] == parameter]
        for method in ("OpenMx", "NPE"):
            subset = (
                at_parameter.loc[at_parameter["method"] == method]
                .set_index("N_pairs")
                .reindex(n_values)
            )
            color = METHOD_COLORS[method]
            axis.fill_between(
                n_values,
                subset["uncertainty_mean_ci_lo"],
                subset["uncertainty_mean_ci_hi"],
                color=color,
                alpha=0.14,
            )
            axis.plot(
                n_values,
                subset["mean_uncertainty"],
                marker="o",
                color=color,
                label="OpenMx SE" if method == "OpenMx" else "NPE posterior SD",
            )
            last_value = float(subset["mean_uncertainty"].iloc[-1])
            axis.annotate(
                f"{last_value:.5f}",
                xy=(last_n, last_value),
                xytext=(0, 10 if method == "OpenMx" else -16),
                textcoords="offset points",
                ha="right",
                color=color,
                fontsize=9,
            )
        axis.set_ylim(bottom=0.0)
        axis.set_title(parameter)
        axis.set_ylabel("Mean reported uncertainty")
        configure_n_axis(axis, n_values, xlabel="Twin-pair sample size N")
    axes[0, 0].legend()
    fig.suptitle(
        "OpenMx SE versus NPE posterior SD "
        f"(test set size = {test_set_size:,} per N)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_coverage_plot(
    metrics: pd.DataFrame,
    n_values: tuple[int, ...],
    path: Path,
) -> None:
    """Compare empirical 95% interval coverage against the nominal target."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), squeeze=False)
    last_n = n_values[-1]
    test_set_size = int(metrics["n_total"].max())
    for col, parameter in enumerate(ACE_PARAM_NAMES):
        axis = axes[0, col]
        at_parameter = metrics.loc[metrics["parameter"] == parameter]
        lower_limit = max(
            0.0,
            float(at_parameter["coverage_95_ci_lo"].min()) - 0.03,
        )
        axis.axhline(
            0.95,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Nominal 95%",
        )
        for method in ("OpenMx", "NPE"):
            subset = (
                at_parameter.loc[at_parameter["method"] == method]
                .set_index("N_pairs")
                .reindex(n_values)
            )
            color = METHOD_COLORS[method]
            axis.fill_between(
                n_values,
                subset["coverage_95_ci_lo"],
                subset["coverage_95_ci_hi"],
                color=color,
                alpha=0.14,
            )
            axis.plot(
                n_values,
                subset["coverage_95"],
                marker="o",
                color=color,
                label=method,
            )
            last_value = float(subset["coverage_95"].iloc[-1])
            axis.annotate(
                f"{last_value:.3f}",
                xy=(last_n, last_value),
                xytext=(0, 10 if method == "OpenMx" else -16),
                textcoords="offset points",
                ha="right",
                color=color,
                fontsize=9,
            )
        axis.set_ylim(lower_limit, 1.005)
        axis.set_title(parameter)
        axis.set_ylabel("Empirical 95% coverage")
        configure_n_axis(axis, n_values, xlabel="Twin-pair sample size N")
    axes[0, 0].legend()
    fig.suptitle(
        "OpenMx versus NPE 95% interval coverage "
        f"(test set size = {test_set_size:,} per N)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def posterior_shape_summary(
    npe: pd.DataFrame,
    n_values: tuple[int, ...],
) -> pd.DataFrame:
    """Summarize per-dataset posterior skewness and excess kurtosis."""
    rows = []
    for n_pairs in n_values:
        at_n = npe.loc[npe["N_pairs"] == n_pairs]
        for parameter in ACE_PARAM_NAMES:
            for statistic in ("skewness", "kurtosis"):
                source_statistic = (
                    "excess_kurtosis" if statistic == "kurtosis" else statistic
                )
                values = at_n[
                    f"{parameter}_npe_{source_statistic}"
                ].to_numpy(dtype=float)
                if statistic == "kurtosis":
                    values = values + 3.0
                values = values[np.isfinite(values)]
                rows.append(
                    {
                        "N_pairs": n_pairs,
                        "parameter": parameter,
                        "statistic": statistic,
                        "n_test_sets": len(values),
                        "mean": float(values.mean()),
                        "median": float(np.median(values)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                    }
                )
    return pd.DataFrame(rows)


def save_posterior_shape_plot(
    npe: pd.DataFrame,
    n_values: tuple[int, ...],
    statistic: str,
    path: Path,
    n_draws: int,
) -> None:
    """Plot posterior shape statistics across test datasets as box plots."""
    labels = {
        "skewness": ("Posterior skewness", "NPE posterior skewness", 0.0),
        "kurtosis": (
            "Posterior kurtosis",
            "NPE posterior kurtosis",
            3.0,
        ),
    }
    ylabel, title, normal_value = labels[statistic]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), squeeze=False)
    positions = np.arange(len(n_values))
    n_test_sets = int(npe.groupby("N_pairs").size().max())
    for col, parameter in enumerate(ACE_PARAM_NAMES):
        axis = axes[0, col]
        source_statistic = (
            "excess_kurtosis" if statistic == "kurtosis" else statistic
        )
        distributions = []
        for n_pairs in n_values:
            values = npe.loc[
                npe["N_pairs"] == n_pairs,
                f"{parameter}_npe_{source_statistic}",
            ].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if statistic == "kurtosis":
                values = values + 3.0
            distributions.append(values)
        axis.axhline(
            normal_value,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"Normal = {normal_value:g}",
        )
        boxes = axis.boxplot(
            distributions,
            positions=positions,
            widths=0.58,
            patch_artist=True,
            showfliers=True,
            boxprops={"edgecolor": METHOD_COLORS["NPE"], "linewidth": 1.2},
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"color": METHOD_COLORS["NPE"]},
            capprops={"color": METHOD_COLORS["NPE"]},
            flierprops={
                "marker": ".",
                "markersize": 2,
                "markerfacecolor": METHOD_COLORS["NPE"],
                "markeredgecolor": METHOD_COLORS["NPE"],
                "alpha": 0.18,
            },
        )
        for box in boxes["boxes"]:
            box.set_facecolor(METHOD_COLORS["NPE"])
            box.set_alpha(0.28)
        if statistic == "kurtosis":
            axis.set_yscale("log")
        axis.set_title(parameter)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, [str(n) for n in n_values])
        axis.set_xlabel("Twin-pair sample size N")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    if statistic == "kurtosis":
        figure_title = f"{title} ({n_draws:,} posterior draws)"
    else:
        figure_title = (
            f"{title} (box plots across {n_test_sets:,} test datasets per N)"
        )
    fig.suptitle(figure_title)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_sbc_rank_plots(
    npe: pd.DataFrame,
    n_values: tuple[int, ...],
    n_draws: int,
    output_dir: Path,
    n_bins: int = 10,
    reference_probability: float = 0.99,
) -> None:
    """Save Talts et al.-style SBC rank histograms for every fixed-N NPE.

    The L + 1 possible integer ranks are split into nearly equal contiguous
    groups. Because L + 1 need not be divisible by n_bins, every gray reference
    interval uses that bin's exact probability under discrete uniform ranks.
    """
    if n_bins < 2 or n_bins > n_draws + 1:
        raise ValueError("n_bins must be between 2 and n_draws + 1")
    if not 0.0 < reference_probability < 1.0:
        raise ValueError("reference_probability must be between 0 and 1")

    rank_groups = np.array_split(np.arange(n_draws + 1), n_bins)
    rank_to_bin = np.empty(n_draws + 1, dtype=int)
    bin_probabilities = np.empty(n_bins, dtype=float)
    for bin_index, group in enumerate(rank_groups):
        rank_to_bin[group] = bin_index
        bin_probabilities[bin_index] = len(group) / (n_draws + 1)

    tail_probability = (1.0 - reference_probability) / 2.0
    x = np.arange(n_bins)
    x_tick_locations = np.linspace(-0.5, n_bins - 0.5, 6)
    x_tick_labels = [f"{value:.1f}" for value in np.linspace(0.0, 1.0, 6)]

    for n_pairs in n_values:
        at_n = npe.loc[npe["N_pairs"] == n_pairs].copy()
        if at_n.empty:
            raise ValueError(f"No NPE rank results found for N={n_pairs}")

        expected = len(at_n) * bin_probabilities
        reference_lo = binom.ppf(
            tail_probability, len(at_n), bin_probabilities
        )
        reference_hi = binom.ppf(
            1.0 - tail_probability, len(at_n), bin_probabilities
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
        for parameter_index, parameter in enumerate(ACE_PARAM_NAMES):
            axis = axes[parameter_index]
            ranks = at_n[f"{parameter}_npe_true_rank"].to_numpy(dtype=int)
            if np.any((ranks < 0) | (ranks > n_draws)):
                raise ValueError(
                    f"{parameter} ranks at N={n_pairs} fall outside [0, {n_draws}]"
                )
            counts = np.bincount(rank_to_bin[ranks], minlength=n_bins)

            for bin_index in range(n_bins):
                axis.fill_between(
                    [bin_index - 0.5, bin_index + 0.5],
                    [reference_lo[bin_index]] * 2,
                    [reference_hi[bin_index]] * 2,
                    color="0.85",
                    linewidth=0,
                    zorder=0,
                )
                axis.plot(
                    [bin_index - 0.5, bin_index + 0.5],
                    [expected[bin_index]] * 2,
                    color="0.35",
                    linewidth=1.1,
                    zorder=1,
                )
            axis.bar(
                x,
                counts,
                width=0.86,
                color=METHOD_COLORS["NPE"],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.78,
                zorder=2,
            )
            axis.set_title(parameter)
            axis.set_xlim(-0.5, n_bins - 0.5)
            axis.set_xticks(x_tick_locations, x_tick_labels)
            axis.set_xlabel("Posterior rank percentile of truth")
            axis.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("Number of simulated datasets")
        fig.suptitle(
            "NPE simulation-based calibration "
            f"(Twin-pair sample size N={n_pairs}; M={len(at_n)}, L={n_draws})"
        )
        fig.text(
            0.5,
            0.01,
            "Gray region: 99% binomial reference interval under uniform ranks",
            ha="center",
            fontsize=9,
            color="0.3",
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        fig.savefig(
            output_dir / f"npe_sbc_rank_N{n_pairs}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def save_sbc_ecdf_plots(
    npe: pd.DataFrame,
    n_values: tuple[int, ...],
    n_draws: int,
    output_dir: Path,
    reference_probability: float = 0.99,
) -> None:
    """Save SBC rank ECDF and ECDF-minus-uniform figures for every fixed N."""
    if not 0.0 < reference_probability < 1.0:
        raise ValueError("reference_probability must be between 0 and 1")

    possible_ranks = np.arange(n_draws + 1)
    uniform_cdf = (possible_ranks + 1) / (n_draws + 1)
    x = np.concatenate(([0.0], uniform_cdf))
    expected = x.copy()
    tail_probability = (1.0 - reference_probability) / 2.0

    for n_pairs in n_values:
        at_n = npe.loc[npe["N_pairs"] == n_pairs].copy()
        if at_n.empty:
            raise ValueError(f"No NPE rank results found for N={n_pairs}")
        n_test_sets = len(at_n)

        reference_lo = binom.ppf(
            tail_probability, n_test_sets, uniform_cdf
        ) / n_test_sets
        reference_hi = binom.ppf(
            1.0 - tail_probability, n_test_sets, uniform_cdf
        ) / n_test_sets
        reference_lo = np.concatenate(([0.0], reference_lo))
        reference_hi = np.concatenate(([0.0], reference_hi))

        fig_ecdf, axes_ecdf = plt.subplots(
            1, 3, figsize=(15, 4.8), sharex=True, sharey=True
        )
        fig_difference, axes_difference = plt.subplots(
            1, 3, figsize=(15, 4.8), sharex=True, sharey=True
        )

        for parameter_index, parameter in enumerate(ACE_PARAM_NAMES):
            ranks = at_n[f"{parameter}_npe_true_rank"].to_numpy(dtype=int)
            if np.any((ranks < 0) | (ranks > n_draws)):
                raise ValueError(
                    f"{parameter} ranks at N={n_pairs} fall outside [0, {n_draws}]"
                )
            cumulative_counts = np.bincount(
                ranks, minlength=n_draws + 1
            ).cumsum()
            empirical = np.concatenate(([0.0], cumulative_counts / n_test_sets))

            ecdf_axis = axes_ecdf[parameter_index]
            ecdf_axis.fill_between(
                x,
                reference_lo,
                reference_hi,
                color="0.85",
                linewidth=0,
                label="99% reference interval",
            )
            ecdf_axis.plot(
                x,
                expected,
                color="0.25",
                linestyle="--",
                linewidth=1.2,
                label="Uniform expectation",
            )
            ecdf_axis.step(
                x,
                empirical,
                where="post",
                color=METHOD_COLORS["NPE"],
                linewidth=1.8,
                label="Empirical rank CDF",
            )
            ecdf_axis.set_title(parameter)
            ecdf_axis.set_xlim(0.0, 1.0)
            ecdf_axis.set_ylim(0.0, 1.0)
            ecdf_axis.set_xlabel("Uniform rank CDF")
            ecdf_axis.grid(alpha=0.2)

            difference_axis = axes_difference[parameter_index]
            difference_axis.fill_between(
                x,
                reference_lo - expected,
                reference_hi - expected,
                color="0.85",
                linewidth=0,
                label="99% reference interval",
            )
            difference_axis.axhline(
                0.0,
                color="0.25",
                linestyle="--",
                linewidth=1.2,
            )
            difference_axis.step(
                x,
                empirical - expected,
                where="post",
                color=METHOD_COLORS["NPE"],
                linewidth=1.8,
                label="Empirical minus uniform CDF",
            )
            difference_axis.set_title(parameter)
            difference_axis.set_xlim(0.0, 1.0)
            difference_axis.set_xlabel("Uniform rank CDF")
            difference_axis.grid(alpha=0.2)

        axes_ecdf[0].set_ylabel("Empirical rank CDF")
        axes_ecdf[0].legend(loc="upper left")
        fig_ecdf.suptitle(
            "NPE SBC rank ECDF "
            f"(Twin-pair sample size N={n_pairs}; M={n_test_sets}, L={n_draws})"
        )
        fig_ecdf.tight_layout(rect=(0, 0, 1, 0.94))
        fig_ecdf.savefig(
            output_dir / f"npe_sbc_ecdf_N{n_pairs}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_ecdf)

        axes_difference[0].set_ylabel("Empirical CDF - uniform CDF")
        axes_difference[0].legend(loc="lower left")
        fig_difference.suptitle(
            "NPE SBC rank ECDF deviation "
            f"(Twin-pair sample size N={n_pairs}; M={n_test_sets}, L={n_draws})"
        )
        fig_difference.tight_layout(rect=(0, 0, 1, 0.94))
        fig_difference.savefig(
            output_dir / f"npe_sbc_ecdf_deviation_N{n_pairs}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_difference)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare OpenMx and fixed-N Dirichlet NPEs on identical data"
    )
    parser.add_argument("--n_conditions", type=int, default=200)
    parser.add_argument(
        "--n_pairs", type=int, nargs="+", default=list(DEFAULT_N_PAIRS)
    )
    parser.add_argument("--n_posterior_samples", type=int, default=2000)
    parser.add_argument(
        "--sbc_bins",
        type=int,
        default=None,
        help="SBC histogram bins; default chooses about 20 test datasets per bin",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=5000,
        help="Paired bootstrap resamples for the bias-vector norm CI",
    )
    parser.add_argument("--dirichlet_alpha", type=float, nargs=3, default=[1, 1, 1])
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--models_dir", default="prior_comparison")
    parser.add_argument("--output_dir", default="dirichlet_openmx_comparison")
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument(
        "--openmx_backend",
        default=str(Path(__file__).with_name("08_fit_openmx_paired_dirichlet.R")),
    )
    parser.add_argument(
        "--reuse_data", action="store_true",
        help="Reuse paired_test_data.csv instead of simulating it again",
    )
    parser.add_argument(
        "--reuse_openmx", action="store_true",
        help="Reuse openmx_estimates.csv instead of refitting OpenMx",
    )
    args = parser.parse_args()

    if args.n_conditions < 2:
        parser.error("--n_conditions must be at least 2")
    if args.n_posterior_samples < 2:
        parser.error("--n_posterior_samples must be at least 2")
    if args.sbc_bins is not None and args.sbc_bins < 2:
        parser.error("--sbc_bins must be at least 2")
    if args.n_bootstrap < 100:
        parser.error("--n_bootstrap must be at least 100")
    n_values = tuple(dict.fromkeys(int(n) for n in args.n_pairs))
    if any(n < 2 for n in n_values):
        parser.error("--n_pairs values must be at least 2")
    alpha = np.asarray(args.dirichlet_alpha, dtype=float)
    if alpha.shape != (3,) or np.any(alpha <= 0):
        parser.error("--dirichlet_alpha must contain three positive values")
    sbc_bins = (
        args.sbc_bins
        if args.sbc_bins is not None
        else max(2, int(round(args.n_conditions / 20)))
    )
    if sbc_bins > args.n_posterior_samples + 1:
        parser.error("--sbc_bins cannot exceed n_posterior_samples + 1")

    models_dir = resolve(args.models_dir, MODELS_DIR)
    output_dir = resolve(args.output_dir, RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = output_dir / "paired_test_data.csv"
    openmx_path = output_dir / "openmx_estimates.csv"
    npe_path = output_dir / "npe_posterior_summaries.csv"
    merged_path = output_dir / "paired_estimates.csv"
    metrics_path = output_dir / "paired_metrics.csv"
    differences_path = output_dir / "paired_estimator_differences.csv"
    bias_norms_path = output_dir / "paired_bias_vector_norm.csv"
    posterior_shape_path = output_dir / "npe_posterior_shape_summary.csv"
    config_path = output_dir / "config.json"

    validate_models(models_dir, n_values)
    if args.reuse_data:
        if not paired_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing {paired_path}")
        paired = pd.read_csv(paired_path)
    else:
        print(
            f"Generating {args.n_conditions} Dirichlet conditions at "
            f"N={list(n_values)} ..."
        )
        paired = generate_paired_data(args.n_conditions, n_values, alpha, args.seed)
        paired.to_csv(paired_path, index=False)
    check_paired_data(paired, args.n_conditions, n_values)
    paired_fingerprint = file_sha256(paired_path)

    if args.reuse_openmx:
        if not openmx_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing {openmx_path}")
        if not config_path.exists():
            raise FileNotFoundError(
                "Cannot safely reuse OpenMx results without the previous config.json"
            )
        with open(config_path) as handle:
            previous_config = json.load(handle)
        if previous_config.get("paired_data_sha256") != paired_fingerprint:
            raise ValueError(
                "The current paired_test_data.csv differs from the dataset used "
                "for openmx_estimates.csv; rerun without --reuse_openmx"
            )
    else:
        run_openmx(
            args.rscript,
            Path(args.openmx_backend).resolve(),
            paired_path.resolve(),
            openmx_path.resolve(),
        )
    openmx = pd.read_csv(openmx_path)

    npe = evaluate_npe(
        paired,
        models_dir,
        n_values,
        args.n_posterior_samples,
        args.seed,
    )
    npe.to_csv(npe_path, index=False)
    merged = merge_results(paired, npe, openmx)
    merged.to_csv(merged_path, index=False)
    metrics, differences = metric_tables(merged, n_values)
    bias_norms = bias_norm_table(
        merged, n_values, args.n_bootstrap, args.seed
    )
    posterior_shape = posterior_shape_summary(npe, n_values)
    metrics.to_csv(metrics_path, index=False)
    differences.to_csv(differences_path, index=False)
    bias_norms.to_csv(bias_norms_path, index=False)
    posterior_shape.to_csv(posterior_shape_path, index=False)

    save_bias_plot(metrics, n_values, output_dir / "paired_bias_by_n.png")
    save_mae_plot(metrics, n_values, output_dir / "paired_mae_by_n.png")
    save_performance_plot(metrics, n_values, output_dir / "paired_performance_by_n.png")
    save_mean_uncertainty_plot(
        metrics, n_values, output_dir / "paired_mean_uncertainty_by_n.png"
    )
    save_coverage_plot(metrics, n_values, output_dir / "paired_coverage_by_n.png")
    save_posterior_shape_plot(
        npe,
        n_values,
        "skewness",
        output_dir / "npe_posterior_skewness_by_n.png",
        args.n_posterior_samples,
    )
    save_posterior_shape_plot(
        npe,
        n_values,
        "kurtosis",
        output_dir / "npe_posterior_kurtosis_by_n.png",
        args.n_posterior_samples,
    )
    save_sbc_rank_plots(
        npe,
        n_values,
        args.n_posterior_samples,
        output_dir,
        n_bins=sbc_bins,
    )
    save_sbc_ecdf_plots(
        npe,
        n_values,
        args.n_posterior_samples,
        output_dir,
    )
    with open(config_path, "w") as handle:
        json.dump(
            {
                "design": "paired_dirichlet_openmx_vs_fixed_n_npe",
                "n_conditions": args.n_conditions,
                "n_pairs": list(n_values),
                "n_posterior_samples": args.n_posterior_samples,
                "dirichlet_alpha": alpha.tolist(),
                "seed": args.seed,
                "models_dir": str(models_dir),
                "n_bootstrap_bias_norm": args.n_bootstrap,
                "paired_data_sha256": paired_fingerprint,
                "metrics_use_common_openmx_converged_rows": True,
                "n_is_npe_feature": False,
                "posterior_kurtosis": "Fisher excess kurtosis",
                "true_rank": "number of posterior draws below truth",
                "sbc_rank_bins": sbc_bins,
                "sbc_rank_reference_probability": 0.99,
                "sbc_rank_rows": "all NPE evaluations, regardless of OpenMx convergence",
                "sbc_ecdf_reference_probability": 0.99,
                "sbc_ecdf_reference_interval": "pointwise binomial under discrete uniform ranks",
                "mean_uncertainty_plot": "OpenMx standard error versus NPE posterior standard deviation",
                "coverage_interval": "95% Wilson score interval across test datasets",
                "posterior_shape_summary": "box plots across NPE test datasets; Pearson kurtosis",
            },
            handle,
            indent=2,
        )

    convergence = (
        openmx.groupby("N_pairs")["openmx_converged"].mean().reindex(n_values)
    )
    print(f"\nSaved paired comparison -> {output_dir}")
    print("OpenMx convergence rate by N:")
    print(convergence.to_string(float_format=lambda value: f"{value:.3f}"))
    print("\nBias and RMSE:")
    print(
        metrics[["method", "N_pairs", "parameter", "bias", "rmse", "coverage_95"]]
        .to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )


if __name__ == "__main__":
    main()
