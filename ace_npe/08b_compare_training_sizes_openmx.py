"""STEP 08b -- compare OpenMx with NPEs trained at four set sizes.

This script consumes the final STEP 03b per-test posterior summaries.  It
regenerates the exact same M covariance datasets from STEP 03b's recorded seed,
fits OpenMx to those covariances, and compares five methods on common usable
rows within every N:

    OpenMx, NPE-100k, NPE-200k, NPE-300k, and NPE-500k.

No NPE is retrained or loaded.  The comparison is possible because STEP 03b
retains posterior summaries for each test dataset even though it discards the
transient fitted models and raw posterior draws.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_model import ACE_PARAM_NAMES, RESULTS_DIR, resolve


SCRIPT_DIR = Path(__file__).resolve().parent
STEP_03B = runpy.run_path(str(SCRIPT_DIR / "03b_training_budget_grid.py"))
DEFAULT_K_VALUES = (100_000, 200_000, 300_000, 500_000)
METHOD_COLORS = {
    "OpenMx": "#1f4e79",
    "NPE-100k": "#e68613",
    "NPE-200k": "#2a9d55",
    "NPE-300k": "#8e63b6",
    "NPE-500k": "#c44e52",
}
METHOD_MARKERS = {
    "OpenMx": "o",
    "NPE-100k": "s",
    "NPE-200k": "^",
    "NPE-300k": "D",
    "NPE-500k": "P",
}


def method_for_k(k_value: int) -> str:
    if k_value % 1_000 == 0:
        return f"NPE-{k_value // 1_000}k"
    return f"NPE-{k_value:,}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def wilson_interval(
    successes: int,
    total: int,
    z_value: float = 1.96,
) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1.0 + z_value**2 / total
    center = (proportion + z_value**2 / (2.0 * total)) / denominator
    half_width = (
        z_value
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z_value**2 / (4.0 * total**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def load_grid_inputs(
    grid_dir: Path,
    requested_k: tuple[int, ...],
) -> tuple[dict, pd.DataFrame, tuple[int, ...]]:
    config_path = grid_dir / "config.json"
    summaries_path = grid_dir / "training_budget_test_summaries.csv"
    complete_path = grid_dir / "COMPLETE"
    if not config_path.exists() or not summaries_path.exists():
        raise FileNotFoundError(
            f"STEP 03b outputs are incomplete in {grid_dir}; expected "
            "config.json and training_budget_test_summaries.csv"
        )
    if not complete_path.exists():
        raise FileNotFoundError(
            f"Missing {complete_path}; wait for STEP 03b aggregation to finish"
        )
    with open(config_path) as handle:
        config = json.load(handle)
    if int(config.get("H", -1)) != 1:
        raise ValueError("STEP 08b expects H=1 in the STEP 03b results")
    available_k = tuple(int(value) for value in config.get("K_values", ()))
    missing_k = sorted(set(requested_k).difference(available_k))
    if missing_k:
        raise ValueError(
            f"STEP 03b results do not contain K={missing_k}; rerun the expanded "
            "49-cell STEP 03b grid before STEP 08b"
        )
    n_values = tuple(int(value) for value in config.get("N_values", ()))
    if not n_values:
        raise ValueError("STEP 03b config has no N_values")

    summaries = pd.read_csv(summaries_path)
    required = {
        "K",
        "N",
        "model_index_h",
        "test_dataset_m",
        "parameter",
        "truth",
        "posterior_mean",
        "posterior_sd",
        "ci_2_5",
        "ci_97_5",
    }
    missing_columns = sorted(required.difference(summaries.columns))
    if missing_columns:
        raise ValueError(
            f"STEP 03b summaries are missing columns: {missing_columns}"
        )
    summaries = summaries.loc[summaries["K"].isin(requested_k)].copy()
    summaries["K"] = summaries["K"].astype(int)
    summaries["N"] = summaries["N"].astype(int)
    summaries["test_dataset_m"] = summaries["test_dataset_m"].astype(int)
    return config, summaries, n_values


def regenerate_paired_data(
    config: dict,
    n_values: tuple[int, ...],
) -> pd.DataFrame:
    """Regenerate the exact STEP 03b test covariance matrices."""
    n_test_datasets = int(config["M"])
    base_seed = int(config["seed"])
    frames = []
    for n_value in n_values:
        test_seed = STEP_03B["seed_from"](base_seed, n_value, 20)
        _, ace, covariance_data = STEP_03B["simulate_fixed_n"](
            n_test_datasets,
            n_value,
            test_seed,
            return_covariances=True,
        )
        frame = covariance_data.copy()
        frame.insert(0, "condition_id", np.arange(1, n_test_datasets + 1))
        frame.insert(1, "N_pairs", n_value)
        for index, parameter in enumerate(ACE_PARAM_NAMES):
            frame[f"{parameter}_true"] = ace[:, index]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validate_pairing(
    summaries: pd.DataFrame,
    paired: pd.DataFrame,
    k_values: tuple[int, ...],
    n_values: tuple[int, ...],
    n_test_datasets: int,
) -> None:
    expected_rows = (
        len(k_values)
        * len(n_values)
        * n_test_datasets
        * len(ACE_PARAM_NAMES)
    )
    if len(summaries) != expected_rows:
        raise ValueError(
            f"Selected STEP 03b summaries have {len(summaries):,} rows; "
            f"expected {expected_rows:,}"
        )
    counts = summaries.groupby(["K", "N", "parameter"]).size()
    if not (counts == n_test_datasets).all():
        raise ValueError("Every (K,N,parameter) must contain exactly M test rows")
    if set(summaries["K"]) != set(k_values):
        raise ValueError("STEP 03b summaries do not contain every requested K")
    if set(summaries["N"]) != set(n_values):
        raise ValueError("STEP 03b summaries do not contain every requested N")

    truth_long = paired.melt(
        id_vars=["condition_id", "N_pairs"],
        value_vars=[f"{parameter}_true" for parameter in ACE_PARAM_NAMES],
        var_name="parameter",
        value_name="regenerated_truth",
    )
    truth_long["parameter"] = truth_long["parameter"].str.replace(
        "_true", "", regex=False
    )
    unique_truth = summaries[
        ["N", "test_dataset_m", "parameter", "truth"]
    ].drop_duplicates()
    checked = unique_truth.merge(
        truth_long,
        left_on=["N", "test_dataset_m", "parameter"],
        right_on=["N_pairs", "condition_id", "parameter"],
        how="left",
        validate="one_to_one",
    )
    if checked["regenerated_truth"].isna().any():
        raise ValueError("Some STEP 03b test truths could not be regenerated")
    maximum_difference = np.max(
        np.abs(checked["truth"] - checked["regenerated_truth"])
    )
    if maximum_difference > 1e-7:
        raise ValueError(
            "Regenerated test data do not match STEP 03b; maximum truth "
            f"difference is {maximum_difference:g}"
        )


def run_openmx(
    rscript: str,
    backend: Path,
    paired_path: Path,
    output_path: Path,
) -> None:
    if not backend.exists():
        raise FileNotFoundError(f"OpenMx backend not found: {backend}")
    command = [rscript, str(backend), str(paired_path), str(output_path)]
    with open(paired_path) as handle:
        row_count = sum(1 for _ in handle) - 1
    print(f"Running OpenMx on {row_count:,} datasets ...", flush=True)
    subprocess.run(command, check=True)


def build_paired_estimates(
    paired: pd.DataFrame,
    openmx: pd.DataFrame,
    summaries: pd.DataFrame,
    k_values: tuple[int, ...],
) -> pd.DataFrame:
    keys = ["condition_id", "N_pairs"]
    if openmx.duplicated(keys).any():
        raise ValueError("OpenMx output contains duplicate condition/N rows")
    base = paired.merge(openmx, on=keys, how="left", validate="one_to_one")
    if base["openmx_converged"].isna().any():
        raise ValueError("Some paired rows have no OpenMx result")

    frames = []
    for n_value, at_n in base.groupby("N_pairs", sort=True):
        finite_estimates = np.isfinite(
            at_n[[f"{parameter}_openmx_est" for parameter in ACE_PARAM_NAMES]]
        ).all(axis=1)
        eligible_ids = set(
            at_n.loc[
                at_n["openmx_converged"].eq(1) & finite_estimates,
                "condition_id",
            ].astype(int)
        )
        if not eligible_ids:
            raise RuntimeError(f"OpenMx had no usable fits at N={n_value}")

        eligible = at_n.loc[at_n["condition_id"].isin(eligible_ids)].copy()
        for parameter in ACE_PARAM_NAMES:
            truth = eligible[f"{parameter}_true"].to_numpy(dtype=float)
            estimate = eligible[f"{parameter}_openmx_est"].to_numpy(dtype=float)
            uncertainty = eligible[f"{parameter}_openmx_se"].to_numpy(dtype=float)
            lower = estimate - 1.96 * uncertainty
            upper = estimate + 1.96 * uncertainty
            frames.append(
                pd.DataFrame(
                    {
                        "N": int(n_value),
                        "test_dataset_m": eligible["condition_id"].to_numpy(
                            dtype=int
                        ),
                        "parameter": parameter,
                        "method": "OpenMx",
                        "K": pd.Series([pd.NA] * len(eligible), dtype="Int64"),
                        "truth": truth,
                        "estimate": estimate,
                        "uncertainty": uncertainty,
                        "ci_lower": lower,
                        "ci_upper": upper,
                    }
                )
            )

        for k_value in k_values:
            selected = summaries.loc[
                (summaries["N"] == n_value)
                & (summaries["K"] == k_value)
                & summaries["test_dataset_m"].isin(eligible_ids)
            ].copy()
            if len(selected) != len(eligible_ids) * len(ACE_PARAM_NAMES):
                raise ValueError(
                    f"Incomplete NPE rows for K={k_value}, N={n_value}"
                )
            frames.append(
                pd.DataFrame(
                    {
                        "N": selected["N"].to_numpy(dtype=int),
                        "test_dataset_m": selected["test_dataset_m"].to_numpy(
                            dtype=int
                        ),
                        "parameter": selected["parameter"].to_numpy(),
                        "method": method_for_k(k_value),
                        "K": pd.Series(
                            [k_value] * len(selected), dtype="Int64"
                        ),
                        "truth": selected["truth"].to_numpy(dtype=float),
                        "estimate": selected["posterior_mean"].to_numpy(dtype=float),
                        "uncertainty": selected["posterior_sd"].to_numpy(dtype=float),
                        "ci_lower": selected["ci_2_5"].to_numpy(dtype=float),
                        "ci_upper": selected["ci_97_5"].to_numpy(dtype=float),
                    }
                )
            )

    output = pd.concat(frames, ignore_index=True)
    output["error"] = output["estimate"] - output["truth"]
    output["absolute_error"] = output["error"].abs()
    output["interval_valid"] = np.isfinite(output["ci_lower"]) & np.isfinite(
        output["ci_upper"]
    )
    output["covered_95"] = (
        output["interval_valid"]
        & (output["truth"] >= output["ci_lower"])
        & (output["truth"] <= output["ci_upper"])
    )
    return output.sort_values(
        ["N", "parameter", "method", "test_dataset_m"]
    ).reset_index(drop=True)


def summarize_metrics(
    estimates: pd.DataFrame,
    n_test_datasets: int,
) -> pd.DataFrame:
    rows = []
    for (method, n_value, parameter), group in estimates.groupby(
        ["method", "N", "parameter"], sort=False
    ):
        errors = group["error"].to_numpy(dtype=float)
        absolute_errors = np.abs(errors)
        uncertainty = group["uncertainty"].to_numpy(dtype=float)
        interval_group = group.loc[group["interval_valid"]]
        covered = interval_group["covered_95"].to_numpy(dtype=bool)
        count = len(group)

        bias = float(errors.mean())
        bias_se = float(errors.std(ddof=1) / np.sqrt(count))
        mae = float(absolute_errors.mean())
        mae_se = float(absolute_errors.std(ddof=1) / np.sqrt(count))
        finite_uncertainty = uncertainty[np.isfinite(uncertainty)]
        if len(finite_uncertainty) < 2:
            raise RuntimeError(
                f"Fewer than two finite uncertainty values for "
                f"{method}, N={n_value}, {parameter}"
            )
        mean_uncertainty = float(finite_uncertainty.mean())
        uncertainty_se = float(
            finite_uncertainty.std(ddof=1) / np.sqrt(len(finite_uncertainty))
        )
        coverage_lower, coverage_upper = wilson_interval(
            int(covered.sum()), len(covered)
        )
        rows.append(
            {
                "method": method,
                "K": (
                    pd.NA
                    if method == "OpenMx"
                    else int(method.removeprefix("NPE-").removesuffix("k")) * 1_000
                ),
                "N": int(n_value),
                "parameter": parameter,
                "n_total": n_test_datasets,
                "n_paired": count,
                "openmx_convergence_rate": count / n_test_datasets,
                "bias": bias,
                "bias_ci_lo": bias - 1.96 * bias_se,
                "bias_ci_hi": bias + 1.96 * bias_se,
                "mae": mae,
                "mae_ci_lo": max(0.0, mae - 1.96 * mae_se),
                "mae_ci_hi": mae + 1.96 * mae_se,
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mean_uncertainty": mean_uncertainty,
                "uncertainty_mean_ci_lo": max(
                    0.0, mean_uncertainty - 1.96 * uncertainty_se
                ),
                "uncertainty_mean_ci_hi": mean_uncertainty
                + 1.96 * uncertainty_se,
                "coverage_95": float(covered.mean()),
                "n_coverage": len(covered),
                "coverage_95_ci_lo": coverage_lower,
                "coverage_95_ci_hi": coverage_upper,
            }
        )
    output = pd.DataFrame(rows)
    output["K"] = output["K"].astype("Int64")
    return output


def configure_n_axis(axis, n_values: tuple[int, ...]) -> None:
    axis.set_xscale("log")
    axis.set_xticks(n_values, [f"{value:,}" for value in n_values])
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.tick_params(axis="x", labelrotation=35)
    axis.set_xlabel("Twin-pair sample size N")
    axis.grid(alpha=0.22)


def plot_line_metric(
    metrics: pd.DataFrame,
    methods: tuple[str, ...],
    n_values: tuple[int, ...],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    interval_columns: tuple[str, str] | None = None,
    reference: float | None = None,
    lower_zero: bool = False,
    annotate_n: int | None = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), squeeze=False)
    annotation_offsets = np.linspace(28, -28, len(methods))
    for column, parameter in enumerate(ACE_PARAM_NAMES):
        axis = axes[0, column]
        if reference is not None:
            axis.axhline(
                reference,
                color="black",
                linestyle="--",
                linewidth=1.1,
                label=("Nominal 95%" if reference == 0.95 else "Zero"),
            )
        for method_index, method in enumerate(methods):
            subset = (
                metrics.loc[
                    (metrics["method"] == method)
                    & (metrics["parameter"] == parameter)
                ]
                .set_index("N")
                .reindex(n_values)
            )
            values = subset[metric].to_numpy(dtype=float)
            color = METHOD_COLORS[method]
            if interval_columns is not None:
                lower = subset[interval_columns[0]].to_numpy(dtype=float)
                upper = subset[interval_columns[1]].to_numpy(dtype=float)
                axis.fill_between(n_values, lower, upper, color=color, alpha=0.10)
            axis.plot(
                n_values,
                values,
                marker=METHOD_MARKERS[method],
                markersize=4.5,
                linewidth=1.7,
                color=color,
                label=method,
            )
            if annotate_n is not None and annotate_n in n_values:
                target_index = n_values.index(annotate_n)
                target_value = float(values[target_index])
                axis.annotate(
                    f"{target_value:.5f}",
                    xy=(annotate_n, target_value),
                    xytext=(8, annotation_offsets[method_index]),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    color=color,
                    fontsize=8,
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                    },
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6},
                )
        if lower_zero:
            axis.set_ylim(bottom=0.0)
        if metric == "coverage_95":
            minimum = float(metrics["coverage_95_ci_lo"].min())
            axis.set_ylim(max(0.0, minimum - 0.03), 1.005)
        axis.set_title(parameter)
        axis.set_ylabel(ylabel)
        configure_n_axis(axis, n_values)
        if annotate_n is not None and annotate_n in n_values:
            axis.set_xlim(min(n_values) / 1.15, max(n_values) * 1.55)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def split_n_ranges(
    n_values: tuple[int, ...],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Split N values so the smaller high-N metrics remain visible."""
    ranges = (
        ("N < 1,000", tuple(value for value in n_values if value < 1_000)),
        ("N ≥ 1,000", tuple(value for value in n_values if value >= 1_000)),
    )
    return tuple((label, values) for label, values in ranges if values)


def plot_split_grouped_bar_metric(
    metrics: pd.DataFrame,
    methods: tuple[str, ...],
    n_values: tuple[int, ...],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    interval_columns: tuple[str, str],
    lower_zero: bool = False,
) -> None:
    """Draw grouped bars and give the low- and high-N ranges separate scales."""
    n_ranges = split_n_ranges(n_values)
    fig, axes = plt.subplots(
        len(n_ranges),
        len(ACE_PARAM_NAMES),
        figsize=(16, 4.3 * len(n_ranges)),
        squeeze=False,
    )
    bar_width = 0.15
    offsets = (
        np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2
    ) * bar_width

    for row, (range_label, range_values) in enumerate(n_ranges):
        base_positions = np.arange(len(range_values), dtype=float)
        for column, parameter in enumerate(ACE_PARAM_NAMES):
            axis = axes[row, column]
            for method_index, method in enumerate(methods):
                subset = (
                    metrics.loc[
                        (metrics["method"] == method)
                        & (metrics["parameter"] == parameter)
                    ]
                    .set_index("N")
                    .reindex(range_values)
                )
                values = subset[metric].to_numpy(dtype=float)
                lower = subset[interval_columns[0]].to_numpy(dtype=float)
                upper = subset[interval_columns[1]].to_numpy(dtype=float)
                yerr = np.vstack(
                    (
                        np.maximum(values - lower, 0.0),
                        np.maximum(upper - values, 0.0),
                    )
                )
                axis.bar(
                    base_positions + offsets[method_index],
                    values,
                    width=bar_width * 0.92,
                    yerr=yerr,
                    capsize=2.5,
                    color=METHOD_COLORS[method],
                    edgecolor="white",
                    linewidth=0.4,
                    error_kw={"ecolor": "black", "elinewidth": 0.9},
                    label=method,
                )
            if lower_zero:
                axis.set_ylim(bottom=0.0)
            axis.set_title(f"{parameter} — {range_label}")
            axis.set_ylabel(ylabel)
            axis.set_xticks(
                base_positions, [f"{value:,}" for value in range_values]
            )
            axis.set_xlabel("Twin-pair sample size N")
            axis.grid(axis="y", alpha=0.22)
            axis.set_axisbelow(True)

    handles = [
        Patch(facecolor=METHOD_COLORS[method], label=method) for method in methods
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(methods),
        fontsize=9,
        bbox_to_anchor=(0.5, 0.91),
    )
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bar_metric(
    metrics: pd.DataFrame,
    methods: tuple[str, ...],
    n_values: tuple[int, ...],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    interval_columns: tuple[str, str],
    reference: float | None = None,
) -> None:
    """Draw one grouped bar panel per ACE parameter."""
    fig, axes = plt.subplots(1, len(ACE_PARAM_NAMES), figsize=(16, 5.2), squeeze=False)
    base_positions = np.arange(len(n_values), dtype=float)
    bar_width = 0.15
    offsets = (
        np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2
    ) * bar_width

    for column, parameter in enumerate(ACE_PARAM_NAMES):
        axis = axes[0, column]
        for method_index, method in enumerate(methods):
            subset = (
                metrics.loc[
                    (metrics["method"] == method)
                    & (metrics["parameter"] == parameter)
                ]
                .set_index("N")
                .reindex(n_values)
            )
            values = subset[metric].to_numpy(dtype=float)
            lower = subset[interval_columns[0]].to_numpy(dtype=float)
            upper = subset[interval_columns[1]].to_numpy(dtype=float)
            yerr = np.vstack(
                (
                    np.maximum(values - lower, 0.0),
                    np.maximum(upper - values, 0.0),
                )
            )
            axis.bar(
                base_positions + offsets[method_index],
                values,
                width=bar_width * 0.92,
                yerr=yerr,
                capsize=2.2,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.4,
                error_kw={"ecolor": "black", "elinewidth": 0.8},
            )
        if reference is not None:
            axis.axhline(
                reference,
                color="black",
                linestyle="--",
                linewidth=1.0,
            )
        axis.set_ylim(0.0, 1.005 if metric == "coverage_95" else None)
        axis.set_title(parameter)
        axis.set_ylabel(ylabel)
        axis.set_xticks(base_positions, [f"{value:,}" for value in n_values])
        axis.tick_params(axis="x", labelrotation=35)
        axis.set_xlabel("Twin-pair sample size N")
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)

    handles = [
        Patch(facecolor=METHOD_COLORS[method], label=method) for method in methods
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(methods),
        fontsize=9,
        bbox_to_anchor=(0.5, 0.88),
    )
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_split_bias_boxplots(
    estimates: pd.DataFrame,
    methods: tuple[str, ...],
    n_values: tuple[int, ...],
    title: str,
    path: Path,
) -> None:
    """Plot dataset-level signed errors; diamonds show mean error (bias)."""
    n_ranges = split_n_ranges(n_values)
    fig, axes = plt.subplots(
        len(n_ranges),
        len(ACE_PARAM_NAMES),
        figsize=(16, 4.5 * len(n_ranges)),
        squeeze=False,
    )
    box_width = 0.13
    offsets = (
        np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2
    ) * box_width
    annotation_offsets = np.linspace(28, -28, len(methods))

    for row, (range_label, range_values) in enumerate(n_ranges):
        base_positions = np.arange(len(range_values), dtype=float)
        for column, parameter in enumerate(ACE_PARAM_NAMES):
            axis = axes[row, column]
            axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
            for method_index, method in enumerate(methods):
                distributions = []
                positions = []
                for n_index, n_value in enumerate(range_values):
                    values = estimates.loc[
                        (estimates["method"] == method)
                        & (estimates["parameter"] == parameter)
                        & (estimates["N"] == n_value),
                        "error",
                    ].to_numpy(dtype=float)
                    distributions.append(values)
                    positions.append(base_positions[n_index] + offsets[method_index])
                color = METHOD_COLORS[method]
                boxplot = axis.boxplot(
                    distributions,
                    positions=positions,
                    widths=box_width * 0.88,
                    patch_artist=True,
                    showmeans=True,
                    showfliers=True,
                    manage_ticks=False,
                    medianprops={"color": "black", "linewidth": 0.9},
                    meanprops={
                        "marker": "D",
                        "markerfacecolor": "white",
                        "markeredgecolor": "black",
                        "markersize": 3.2,
                    },
                    whiskerprops={"color": color, "linewidth": 0.8},
                    capprops={"color": color, "linewidth": 0.8},
                    flierprops={
                        "marker": ".",
                        "markerfacecolor": color,
                        "markeredgecolor": color,
                        "markersize": 1.4,
                        "alpha": 0.22,
                    },
                )
                for box in boxplot["boxes"]:
                    box.set_facecolor(color)
                    box.set_edgecolor(color)
                    box.set_alpha(0.78)
            if 20_000 in range_values:
                n_index = range_values.index(20_000)
                for method_index, method in enumerate(methods):
                    values = estimates.loc[
                        (estimates["method"] == method)
                        & (estimates["parameter"] == parameter)
                        & (estimates["N"] == 20_000),
                        "error",
                    ].to_numpy(dtype=float)
                    mean_error = float(values.mean())
                    mean_position = (
                        base_positions[n_index] + offsets[method_index]
                    )
                    color = METHOD_COLORS[method]
                    axis.annotate(
                        f"{mean_error:.5f}",
                        xy=(mean_position, mean_error),
                        xytext=(8, annotation_offsets[method_index]),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        color=color,
                        fontsize=7.5,
                        bbox={
                            "boxstyle": "round,pad=0.12",
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.80,
                        },
                        arrowprops={
                            "arrowstyle": "-",
                            "color": color,
                            "lw": 0.6,
                        },
                    )
                axis.set_xlim(-0.5, len(range_values) - 0.5 + 0.8)
            axis.set_title(f"{parameter} — {range_label}")
            axis.set_ylabel("Signed error (estimate - truth)")
            axis.set_xticks(
                base_positions, [f"{value:,}" for value in range_values]
            )
            axis.set_xlabel("Twin-pair sample size N")
            axis.grid(axis="y", alpha=0.22)
            axis.set_axisbelow(True)

    handles = [
        Patch(facecolor=METHOD_COLORS[method], label=method) for method in methods
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
            label="Mean error (bias)",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        fontsize=9,
        bbox_to_anchor=(0.5, 0.91),
    )
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_figures(
    metrics: pd.DataFrame,
    estimates: pd.DataFrame,
    k_values: tuple[int, ...],
    n_values: tuple[int, ...],
    n_test_datasets: int,
    output_dir: Path,
) -> None:
    methods = ("OpenMx",) + tuple(method_for_k(value) for value in k_values)
    suffix = f"(M={n_test_datasets:,} test datasets per N; paired on usable OpenMx fits)"
    plot_split_bias_boxplots(
        estimates,
        methods,
        n_values,
        f"Paired signed-error distributions: OpenMx versus NPE training set sizes\n"
        f"{suffix}; diamonds mark mean error (bias)",
        output_dir / "paired_bias_by_n.png",
    )
    plot_split_grouped_bar_metric(
        metrics,
        methods,
        n_values,
        "mae",
        "Mean absolute error",
        f"Paired mean absolute error: OpenMx versus NPE training set sizes\n{suffix}",
        output_dir / "paired_mae_by_n.png",
        interval_columns=("mae_ci_lo", "mae_ci_hi"),
        lower_zero=True,
    )
    plot_line_metric(
        metrics,
        methods,
        n_values,
        "rmse",
        "RMSE",
        f"Paired RMSE: OpenMx versus NPE training set sizes\n{suffix}",
        output_dir / "paired_rmse_by_n.png",
        lower_zero=True,
        annotate_n=20_000,
    )
    plot_split_grouped_bar_metric(
        metrics,
        methods,
        n_values,
        "mean_uncertainty",
        "Mean uncertainty",
        f"OpenMx SE versus NPE posterior SD\n{suffix}",
        output_dir / "paired_mean_uncertainty_by_n.png",
        interval_columns=("uncertainty_mean_ci_lo", "uncertainty_mean_ci_hi"),
        lower_zero=True,
    )
    plot_grouped_bar_metric(
        metrics,
        methods,
        n_values,
        "coverage_95",
        "Empirical 95% coverage",
        f"Paired 95% interval coverage\n{suffix}",
        output_dir / "paired_coverage_by_n.png",
        interval_columns=("coverage_95_ci_lo", "coverage_95_ci_hi"),
        reference=0.95,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare OpenMx with NPE-100k, NPE-200k, NPE-300k, and NPE-500k"
        )
    )
    parser.add_argument("--training-grid-dir", default="training_budget_grid")
    parser.add_argument(
        "--output-dir", default="training_set_size_openmx_comparison"
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES)
    )
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument(
        "--openmx-backend",
        default=str(SCRIPT_DIR / "08_fit_openmx_paired_dirichlet.R"),
    )
    parser.add_argument(
        "--reuse-openmx",
        action="store_true",
        help="Reuse prior OpenMx fits only if the paired-data fingerprint matches",
    )
    args = parser.parse_args()
    args.k_values = tuple(dict.fromkeys(int(value) for value in args.k_values))
    if any(value <= 0 for value in args.k_values):
        parser.error("all K values must be positive")
    unsupported = [
        method_for_k(value)
        for value in args.k_values
        if method_for_k(value) not in METHOD_COLORS
    ]
    if unsupported:
        parser.error(f"no plot colors configured for methods: {unsupported}")
    return args


def main() -> None:
    args = parse_args()
    grid_dir = resolve(args.training_grid_dir, RESULTS_DIR)
    output_dir = resolve(args.output_dir, RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "COMPLETE").unlink(missing_ok=True)

    grid_config, summaries, n_values = load_grid_inputs(
        grid_dir, args.k_values
    )
    n_test_datasets = int(grid_config["M"])
    paired = regenerate_paired_data(grid_config, n_values)
    validate_pairing(
        summaries,
        paired,
        args.k_values,
        n_values,
        n_test_datasets,
    )

    paired_path = output_dir / "paired_test_data.csv"
    openmx_path = output_dir / "openmx_estimates.csv"
    estimates_path = output_dir / "paired_estimates_long.csv"
    metrics_path = output_dir / "comparison_metrics.csv"
    config_path = output_dir / "config.json"
    write_csv_atomic(paired, paired_path)
    paired_fingerprint = file_sha256(paired_path)
    grid_summaries_fingerprint = file_sha256(
        grid_dir / "training_budget_test_summaries.csv"
    )

    if args.reuse_openmx:
        if not openmx_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                "Cannot reuse OpenMx without openmx_estimates.csv and config.json"
            )
        with open(config_path) as handle:
            previous_config = json.load(handle)
        if previous_config.get("paired_data_sha256") != paired_fingerprint:
            raise ValueError(
                "Paired test data changed; rerun without --reuse-openmx"
            )
    else:
        run_openmx(
            args.rscript,
            Path(args.openmx_backend).resolve(),
            paired_path.resolve(),
            openmx_path.resolve(),
        )

    openmx = pd.read_csv(openmx_path)
    estimates = build_paired_estimates(
        paired, openmx, summaries, args.k_values
    )
    metrics = summarize_metrics(estimates, n_test_datasets)
    write_csv_atomic(estimates, estimates_path)
    write_csv_atomic(metrics, metrics_path)
    save_figures(
        metrics,
        estimates,
        args.k_values,
        n_values,
        n_test_datasets,
        output_dir,
    )

    config = {
        "design": "paired_openmx_vs_multiple_npe_training_set_sizes",
        "training_grid_dir": str(grid_dir),
        "output_dir": str(output_dir),
        "K_values": list(args.k_values),
        "methods": ["OpenMx"]
        + [method_for_k(value) for value in args.k_values],
        "N_values": list(n_values),
        "H": 1,
        "M": n_test_datasets,
        "L": int(grid_config["L"]),
        "paired_within_N": True,
        "metrics_use_common_openmx_converged_rows": True,
        "openmx_uncertainty": "delta-method standard error",
        "npe_uncertainty": "posterior standard deviation",
        "npe_models_retrained": False,
        "paired_data_sha256": paired_fingerprint,
        "training_grid_summaries_sha256": grid_summaries_fingerprint,
        "openmx_backend": str(Path(args.openmx_backend).resolve()),
    }
    write_json_atomic(config, config_path)
    (output_dir / "COMPLETE").write_text("complete\n")
    print(f"Saved paired multi-NPE OpenMx comparison -> {output_dir}")


if __name__ == "__main__":
    main()
