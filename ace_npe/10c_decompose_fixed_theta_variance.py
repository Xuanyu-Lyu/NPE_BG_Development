"""STEP 10c -- decompose fixed-theta posterior-mean variance.

This script reuses the fully crossed STEP 10b evaluation: 100 independently
trained NPEs evaluated on the same 500 fixed-theta datasets.  The datasets are
randomly divided into five blocks of 100.  Within each block and ACE parameter,
the observed posterior mean is represented as

    mean[m, j] = grand mean + model[m] + dataset[j]
                 + remainder[m, j] + posterior-sampling MC error.

A balanced two-way random-effects method-of-moments calculation estimates the
model and dataset variance components.  The remainder primarily contains the
model-by-dataset interaction.  The posterior-mean Monte Carlo variance is
estimated as posterior_variance / n_posterior_draws.  A one-to-one random
model/dataset matching supplies the requested independent estimate of total
variance in every block.

No NPE is retrained and no new posterior samples are required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ace_model import ACE_PARAM_NAMES, COV_FEATURE_NAMES, RESULTS_DIR, resolve


DEFAULT_ENSEMBLE_DIR = "fixed_theta_npe_ensemble_evaluation"
DEFAULT_OUTPUT_DIR = "fixed_theta_variance_decomposition"
DEFAULT_REPLICATES = 100
DEFAULT_BLOCKS = 5
DEFAULT_POSTERIOR_DRAWS = 2_000
DEFAULT_SEED = 202_609_24

COMPONENT_ORDER = [
    "Total",
    "Model",
    "Dataset",
    "Remainder",
    "Posterior-mean MC",
]
COMPONENT_COLORS = {
    "Total": "#4c4c4c",
    "Model": "#4c78a8",
    "Dataset": "#f58518",
    "Remainder": "#54a24b",
    "Posterior-mean MC": "#b279a2",
}
PARAMETER_COLORS = {"A": "#1f77b4", "C": "#ff7f0e", "E": "#2ca02c"}


def load_evaluation(
    ensemble_dir: Path,
    n_replicates: int,
    requested_draws: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, int]:
    """Load and validate the completed, shared-dataset STEP 10b results."""
    means_by_parameter: dict[str, list[np.ndarray]] = {
        parameter: [] for parameter in ACE_PARAM_NAMES
    }
    sds_by_parameter: dict[str, list[np.ndarray]] = {
        parameter: [] for parameter in ACE_PARAM_NAMES
    }
    reference_shared: pd.DataFrame | None = None
    dataset_ids: np.ndarray | None = None
    recorded_draw_counts: set[int] = set()

    for replicate in range(1, n_replicates + 1):
        replicate_dir = ensemble_dir / f"replicate_{replicate:03d}"
        result_path = replicate_dir / "fixed_theta_posterior_results.csv"
        complete_path = replicate_dir / "COMPLETE"
        if not complete_path.exists() or not result_path.exists():
            raise FileNotFoundError(
                f"Missing completed STEP 10b replicate {replicate}: {result_path}"
            )

        config_path = replicate_dir / "config.json"
        if config_path.exists():
            with config_path.open() as handle:
                config = json.load(handle)
            if "n_posterior_draws" in config:
                recorded_draw_counts.add(int(config["n_posterior_draws"]))

        frame = pd.read_csv(result_path).sort_values("test_simulation")
        required = {
            "test_simulation",
            *COV_FEATURE_NAMES,
            *(f"true_{parameter}" for parameter in ACE_PARAM_NAMES),
            *(f"{parameter}_posterior_mean" for parameter in ACE_PARAM_NAMES),
            *(f"{parameter}_posterior_sd" for parameter in ACE_PARAM_NAMES),
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{result_path} is missing columns: {sorted(missing)}")
        if frame["test_simulation"].duplicated().any():
            raise ValueError(f"Duplicate test_simulation values in {result_path}")

        shared_columns = [
            "test_simulation",
            *COV_FEATURE_NAMES,
            *(f"true_{parameter}" for parameter in ACE_PARAM_NAMES),
        ]
        current_shared = frame[shared_columns].reset_index(drop=True)
        if reference_shared is None:
            reference_shared = current_shared
            dataset_ids = current_shared["test_simulation"].to_numpy(dtype=int)
        else:
            if not np.array_equal(
                current_shared["test_simulation"].to_numpy(),
                reference_shared["test_simulation"].to_numpy(),
            ):
                raise ValueError(f"Dataset IDs differ in replicate {replicate}")
            numeric_columns = shared_columns[1:]
            if not np.allclose(
                current_shared[numeric_columns].to_numpy(dtype=float),
                reference_shared[numeric_columns].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-10,
            ):
                raise ValueError(
                    "STEP 10c requires shared test datasets, but replicate "
                    f"{replicate} contains different data"
                )

        for parameter in ACE_PARAM_NAMES:
            means = frame[f"{parameter}_posterior_mean"].to_numpy(dtype=float)
            sds = frame[f"{parameter}_posterior_sd"].to_numpy(dtype=float)
            if not np.isfinite(means).all() or not np.isfinite(sds).all():
                raise ValueError(f"Non-finite {parameter} result in {result_path}")
            if np.any(sds <= 0):
                raise ValueError(f"Non-positive {parameter} posterior SD in {result_path}")
            means_by_parameter[parameter].append(means)
            sds_by_parameter[parameter].append(sds)

    if len(recorded_draw_counts) > 1:
        raise ValueError(
            f"Replicates record different posterior draw counts: {recorded_draw_counts}"
        )
    if recorded_draw_counts and recorded_draw_counts != {requested_draws}:
        raise ValueError(
            f"STEP 10b used {next(iter(recorded_draw_counts))} posterior draws, "
            f"but STEP 10c was given --n_posterior_draws={requested_draws}"
        )
    assert dataset_ids is not None
    means_arrays = {
        parameter: np.vstack(rows) for parameter, rows in means_by_parameter.items()
    }
    sd_arrays = {
        parameter: np.vstack(rows) for parameter, rows in sds_by_parameter.items()
    }
    return means_arrays, sd_arrays, dataset_ids, requested_draws


def make_blocks(
    dataset_ids: np.ndarray,
    n_models: int,
    n_blocks: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Randomly split datasets into blocks permitting one-to-one matching."""
    if len(dataset_ids) != n_models * n_blocks:
        raise ValueError(
            "One-to-one block matching requires n_datasets = n_models * n_blocks; "
            f"observed {len(dataset_ids)} != {n_models} * {n_blocks}"
        )
    positions = rng.permutation(len(dataset_ids))
    return [
        positions[index * n_models : (index + 1) * n_models]
        for index in range(n_blocks)
    ]


def decompose_block(
    values: np.ndarray,
    posterior_sds: np.ndarray,
    n_posterior_draws: int,
) -> dict[str, float]:
    """Estimate crossed random-effects variance components for one block."""
    n_models, n_datasets = values.shape
    if n_models < 2 or n_datasets < 2:
        raise ValueError("Variance decomposition requires at least two rows and columns")

    grand_mean = float(values.mean())
    model_means = values.mean(axis=1)
    dataset_means = values.mean(axis=0)
    residuals = (
        values
        - model_means[:, None]
        - dataset_means[None, :]
        + grand_mean
    )

    ss_model = n_datasets * np.square(model_means - grand_mean).sum()
    ss_dataset = n_models * np.square(dataset_means - grand_mean).sum()
    ss_interaction = np.square(residuals).sum()
    ms_model = ss_model / (n_models - 1)
    ms_dataset = ss_dataset / (n_datasets - 1)
    ms_interaction = ss_interaction / ((n_models - 1) * (n_datasets - 1))

    mc_variance = float(
        np.mean(np.square(posterior_sds) / float(n_posterior_draws))
    )
    model_variance = float((ms_model - ms_interaction) / n_datasets)
    dataset_variance = float((ms_dataset - ms_interaction) / n_models)
    remainder_variance = float(ms_interaction - mc_variance)

    return {
        "Model": model_variance,
        "Dataset": dataset_variance,
        "Remainder": remainder_variance,
        "Posterior-mean MC": mc_variance,
        "component_sum": (
            model_variance
            + dataset_variance
            + remainder_variance
            + mc_variance
        ),
        "crossed_observed_variance": float(values.var(ddof=1)),
        "grand_mean": grand_mean,
    }


def calculate_results(
    means: dict[str, np.ndarray],
    posterior_sds: dict[str, np.ndarray],
    dataset_ids: np.ndarray,
    blocks: list[np.ndarray],
    n_posterior_draws: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate primary block components and detailed conditional diagnostics."""
    component_rows: list[dict] = []
    detail_rows: list[dict] = []
    assignment_rows: list[dict] = []
    point_rows: list[dict] = []
    n_models = next(iter(means.values())).shape[0]

    for block_number, positions in enumerate(blocks, start=1):
        pairing = rng.permutation(len(positions))
        paired_positions = positions[pairing]
        for model_index, dataset_position in enumerate(paired_positions, start=1):
            assignment_rows.append(
                {
                    "block": block_number,
                    "model": model_index,
                    "test_simulation": int(dataset_ids[dataset_position]),
                }
            )

        for parameter in ACE_PARAM_NAMES:
            block_values = means[parameter][:, positions]
            block_sds = posterior_sds[parameter][:, positions]
            paired_values = means[parameter][np.arange(n_models), paired_positions]
            total_paired = float(paired_values.var(ddof=1))
            decomposition = decompose_block(
                block_values, block_sds, n_posterior_draws
            )
            mc_cell_variances = (
                np.square(block_sds) / float(n_posterior_draws)
            )
            grand_mean = block_values.mean()
            model_means = block_values.mean(axis=1)
            dataset_means = block_values.mean(axis=0)
            residuals = (
                block_values
                - model_means[:, None]
                - dataset_means[None, :]
                + grand_mean
            )

            primary_values = {
                "Total": total_paired,
                **{name: decomposition[name] for name in COMPONENT_ORDER[1:]},
            }
            for component, value in primary_values.items():
                component_rows.append(
                    {
                        "block": block_number,
                        "parameter": parameter,
                        "component": component,
                        "variance": value,
                        "n_models": n_models,
                        "n_datasets": len(positions),
                        "n_posterior_draws": n_posterior_draws,
                        "component_sum": decomposition["component_sum"],
                        "crossed_observed_variance": decomposition[
                            "crossed_observed_variance"
                        ],
                        "paired_total_minus_component_sum": (
                            total_paired - decomposition["component_sum"]
                        ),
                    }
                )

            point_rows.append(
                {
                    "block": block_number,
                    "parameter": parameter,
                    "component": "Total",
                    "unit_type": "paired_block",
                    "model": np.nan,
                    "test_simulation": np.nan,
                    "variance": total_paired,
                }
            )

            # These conditional distributions are useful diagnostics but are
            # not the pure random-effects components plotted in the main figures.
            for local_index, dataset_position in enumerate(positions):
                conditional_model_variance = float(
                    block_values[:, local_index].var(ddof=1)
                )
                detail_rows.append(
                    {
                        "block": block_number,
                        "parameter": parameter,
                        "diagnostic": "model_variance_conditional_on_dataset",
                        "unit": int(dataset_ids[dataset_position]),
                        "variance": conditional_model_variance,
                    }
                )
                point_rows.append(
                    {
                        "block": block_number,
                        "parameter": parameter,
                        "component": "Model",
                        "unit_type": "dataset",
                        "model": np.nan,
                        "test_simulation": int(dataset_ids[dataset_position]),
                        "variance": (
                            conditional_model_variance
                            - decomposition["Remainder"]
                            - float(mc_cell_variances[:, local_index].mean())
                        ),
                    }
                )
            for model_index in range(n_models):
                conditional_dataset_variance = float(
                    block_values[model_index, :].var(ddof=1)
                )
                mean_model_mc_variance = float(
                    mc_cell_variances[model_index, :].mean()
                )
                detail_rows.append(
                    {
                        "block": block_number,
                        "parameter": parameter,
                        "diagnostic": "dataset_variance_conditional_on_model",
                        "unit": model_index + 1,
                        "variance": conditional_dataset_variance,
                    }
                )
                detail_rows.append(
                    {
                        "block": block_number,
                        "parameter": parameter,
                        "diagnostic": "mean_MC_variance_within_model",
                        "unit": model_index + 1,
                        "variance": mean_model_mc_variance,
                    }
                )
                point_rows.append(
                    {
                        "block": block_number,
                        "parameter": parameter,
                        "component": "Dataset",
                        "unit_type": "model",
                        "model": model_index + 1,
                        "test_simulation": np.nan,
                        "variance": (
                            conditional_dataset_variance
                            - decomposition["Remainder"]
                            - mean_model_mc_variance
                        ),
                    }
                )
                point_rows.append(
                    {
                        "block": block_number,
                        "parameter": parameter,
                        "component": "Remainder",
                        "unit_type": "model",
                        "model": model_index + 1,
                        "test_simulation": np.nan,
                        "variance": (
                            float(residuals[model_index, :].var(ddof=1))
                            - mean_model_mc_variance
                        ),
                    }
                )
                paired_local_index = int(pairing[model_index])
                paired_dataset_position = int(positions[paired_local_index])
                point_rows.append(
                    {
                        "block": block_number,
                        "parameter": parameter,
                        "component": "Posterior-mean MC",
                        "unit_type": "paired_model_dataset",
                        "model": model_index + 1,
                        "test_simulation": int(
                            dataset_ids[paired_dataset_position]
                        ),
                        "variance": float(
                            mc_cell_variances[model_index, paired_local_index]
                        ),
                    }
                )

    components = pd.DataFrame(component_rows)
    components["component"] = pd.Categorical(
        components["component"], categories=COMPONENT_ORDER, ordered=True
    )
    components = components.sort_values(
        ["parameter", "block", "component"]
    ).reset_index(drop=True)
    details = pd.DataFrame(detail_rows)
    assignments = pd.DataFrame(assignment_rows)
    point_estimates = pd.DataFrame(point_rows)
    point_estimates["component"] = pd.Categorical(
        point_estimates["component"], categories=COMPONENT_ORDER, ordered=True
    )
    point_estimates = point_estimates.sort_values(
        ["parameter", "block", "component"]
    ).reset_index(drop=True)
    return components, details, assignments, point_estimates


def component_summary(
    components: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    summary = (
        components.groupby(["parameter", "component"], observed=True)[value_column]
        .agg(
            n_blocks="count",
            mean="mean",
            sd="std",
            minimum="min",
            median="median",
            maximum="max",
        )
        .reset_index()
    )
    summary["se_across_blocks"] = summary["sd"] / np.sqrt(summary["n_blocks"])
    return summary


def save_bar_plot(
    summary: pd.DataFrame,
    output_path: Path,
    n_blocks: int,
    y_label: str,
    title_label: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=False)
    positions = np.arange(len(COMPONENT_ORDER))
    for axis, parameter in zip(axes, ACE_PARAM_NAMES):
        selected = summary.loc[summary["parameter"] == parameter].set_index(
            "component"
        ).reindex(COMPONENT_ORDER)
        values = selected["mean"].to_numpy(dtype=float)
        errors = selected["sd"].to_numpy(dtype=float)
        axis.bar(
            positions,
            values,
            yerr=errors,
            capsize=4,
            color=[COMPONENT_COLORS[name] for name in COMPONENT_ORDER],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.6,
        )
        axis.axhline(0, color="0.35", linewidth=0.9)
        axis.set_xticks(positions, COMPONENT_ORDER, rotation=28, ha="right")
        axis.set_title(parameter, color=PARAMETER_COLORS[parameter], fontweight="bold")
        axis.set_ylabel(y_label)
        axis.grid(axis="y", alpha=0.22)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    fig.suptitle(
        f"Fixed-theta {title_label}: mean across {n_blocks} blocks\n"
        "Error bars are ±1 SD across blocks"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_boxplot(
    components: pd.DataFrame,
    output_path: Path,
    value_column: str,
    y_label: str,
    title_label: str,
) -> None:
    n_blocks = int(components["block"].nunique())
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=False)
    positions = np.arange(1, len(COMPONENT_ORDER) + 1)
    rng = np.random.default_rng(20260925)
    for axis, parameter in zip(axes, ACE_PARAM_NAMES):
        selected = components.loc[components["parameter"] == parameter]
        values = [
            selected.loc[selected["component"] == name, value_column]
            .dropna()
            .to_numpy()
            for name in COMPONENT_ORDER
        ]
        boxplot_values = [
            component_values
            if len(component_values)
            else np.asarray([np.nan], dtype=float)
            for component_values in values
        ]
        boxes = axis.boxplot(
            boxplot_values,
            positions=positions,
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "0.15", "linewidth": 1.5},
        )
        for box, component in zip(boxes["boxes"], COMPONENT_ORDER):
            box.set_facecolor(COMPONENT_COLORS[component])
            box.set_alpha(0.28)
            box.set_edgecolor(COMPONENT_COLORS[component])
        for position, (component, component_values) in enumerate(
            zip(COMPONENT_ORDER, values), start=1
        ):
            jitter = rng.uniform(-0.10, 0.10, len(component_values))
            axis.scatter(
                position + jitter,
                component_values,
                s=36,
                color=COMPONENT_COLORS[component],
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        axis.axhline(0, color="0.35", linewidth=0.9)
        axis.set_xticks(positions, COMPONENT_ORDER, rotation=28, ha="right")
        axis.set_title(parameter, color=PARAMETER_COLORS[parameter], fontweight="bold")
        axis.set_ylabel(y_label)
        axis.grid(axis="y", alpha=0.22)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    fig.suptitle(
        f"Fixed-theta {title_label} across {n_blocks} dataset blocks\n"
        "Each point is one block-level estimate"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_block_point_plot(
    point_estimates: pd.DataFrame,
    output_path: Path,
    value_column: str,
    y_label: str,
    title_label: str,
) -> None:
    n_blocks = int(point_estimates["block"].nunique())
    non_total_counts = (
        point_estimates.loc[point_estimates["component"] != "Total"]
        .groupby(["parameter", "block", "component"], observed=True)
        .size()
    )
    points_per_component = int(non_total_counts.max())
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=False)
    positions = np.arange(len(COMPONENT_ORDER))
    block_colors = plt.get_cmap("tab20", n_blocks)(np.arange(n_blocks))
    block_offsets = np.linspace(-0.24, 0.24, n_blocks)
    rng = np.random.default_rng(20260926)
    for axis, parameter in zip(axes, ACE_PARAM_NAMES):
        selected = point_estimates.loc[point_estimates["parameter"] == parameter]
        for block_index, (color, block) in enumerate(
            zip(block_colors, sorted(selected["block"].unique()))
        ):
            one_block = selected.loc[selected["block"] == block]
            for component_index, component in enumerate(COMPONENT_ORDER):
                values = one_block.loc[
                    one_block["component"] == component, value_column
                ].dropna().to_numpy(dtype=float)
                if not len(values):
                    continue
                jitter = rng.uniform(-0.035, 0.035, size=len(values))
                is_total = component == "Total"
                axis.scatter(
                    component_index + block_offsets[block_index] + jitter,
                    values,
                    marker="D" if is_total else "o",
                    s=48 if is_total else 13,
                    alpha=0.90 if is_total else 0.30,
                    color=color,
                    edgecolor="white" if is_total else "none",
                    linewidth=0.45 if is_total else 0,
                    label=f"Block {block}" if component_index == 0 else None,
                    zorder=4 if is_total else 3,
                )
        axis.axhline(0, color="0.35", linewidth=0.9)
        axis.set_xticks(positions, COMPONENT_ORDER, rotation=28, ha="right")
        axis.set_title(parameter, color=PARAMETER_COLORS[parameter], fontweight="bold")
        axis.set_ylabel(y_label)
        axis.grid(axis="y", alpha=0.22)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Fixed-theta {title_label} by block\n"
        f"Diamonds: one Total per block; circles: {points_per_component} "
        "estimates per block and component"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble_dir", default=DEFAULT_ENSEMBLE_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n_replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--n_blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument(
        "--n_posterior_draws", type=int, default=DEFAULT_POSTERIOR_DRAWS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if min(args.n_replicates, args.n_blocks, args.n_posterior_draws) < 1:
        parser.error("replicates, blocks, and posterior draws must be positive")
    return args


def main() -> None:
    args = parse_args()
    ensemble_dir = resolve(args.ensemble_dir, RESULTS_DIR)
    output_dir = resolve(args.output_dir, RESULTS_DIR)
    if not ensemble_dir.is_dir():
        raise FileNotFoundError(f"STEP 10b evaluation directory not found: {ensemble_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    means, posterior_sds, dataset_ids, n_draws = load_evaluation(
        ensemble_dir, args.n_replicates, args.n_posterior_draws
    )
    rng = np.random.default_rng(args.seed)
    blocks = make_blocks(dataset_ids, args.n_replicates, args.n_blocks, rng)
    components, details, assignments, point_estimates = calculate_results(
        means,
        posterior_sds,
        dataset_ids,
        blocks,
        n_draws,
        rng,
    )
    components["standard_error"] = np.nan
    nonnegative = components["variance"] >= 0
    components.loc[nonnegative, "standard_error"] = np.sqrt(
        components.loc[nonnegative, "variance"]
    )
    point_estimates["standard_error"] = np.nan
    nonnegative_points = point_estimates["variance"] >= 0
    point_estimates.loc[nonnegative_points, "standard_error"] = np.sqrt(
        point_estimates.loc[nonnegative_points, "variance"]
    )
    variance_summary = component_summary(components, "variance")
    se_summary = component_summary(components, "standard_error")

    components_path = output_dir / "variance_components_by_block.csv"
    variance_summary_path = output_dir / "variance_component_summary.csv"
    se_summary_path = output_dir / "se_component_summary.csv"
    details_path = output_dir / "conditional_variance_diagnostics.csv"
    point_estimates_path = output_dir / "component_point_estimates.csv"
    assignments_path = output_dir / "block_pairing_assignments.csv"
    components.to_csv(components_path, index=False)
    variance_summary.to_csv(variance_summary_path, index=False)
    se_summary.to_csv(se_summary_path, index=False)
    details.to_csv(details_path, index=False)
    point_estimates.to_csv(point_estimates_path, index=False)
    assignments.to_csv(assignments_path, index=False)

    variance_bar_path = output_dir / "variance_components_mean_bar.png"
    variance_boxplot_path = output_dir / "variance_components_boxplot.png"
    variance_points_path = output_dir / "variance_components_block_points.png"
    save_bar_plot(
        variance_summary,
        variance_bar_path,
        args.n_blocks,
        "Variance",
        "variance decomposition",
    )
    save_boxplot(
        components,
        variance_boxplot_path,
        "variance",
        "Variance",
        "variance decomposition",
    )
    save_block_point_plot(
        point_estimates,
        variance_points_path,
        "variance",
        "Variance",
        "variance components",
    )

    se_bar_path = output_dir / "se_components_mean_bar.png"
    se_boxplot_path = output_dir / "se_components_boxplot.png"
    se_points_path = output_dir / "se_components_block_points.png"
    save_bar_plot(
        se_summary,
        se_bar_path,
        args.n_blocks,
        "SE = sqrt(variance)",
        "SE-scale decomposition",
    )
    save_boxplot(
        components,
        se_boxplot_path,
        "standard_error",
        "SE = sqrt(variance)",
        "SE-scale decomposition",
    )
    save_block_point_plot(
        point_estimates,
        se_points_path,
        "standard_error",
        "SE = sqrt(variance)",
        "SE-scale components",
    )

    negative = components.loc[
        (components["component"] != "Total")
        & (components["variance"] < 0),
        ["block", "parameter", "component", "variance"],
    ]
    config = {
        "experiment": "fixed_theta_variance_decomposition",
        "source_ensemble_dir": str(ensemble_dir),
        "n_models": args.n_replicates,
        "n_datasets": int(len(dataset_ids)),
        "n_blocks": args.n_blocks,
        "datasets_per_block": int(len(dataset_ids) // args.n_blocks),
        "n_posterior_draws": n_draws,
        "seed": args.seed,
        "parameters": list(ACE_PARAM_NAMES),
        "method": "balanced crossed random-effects method of moments",
        "total_variance_method": (
            "sample variance of a random one-to-one model/dataset pairing "
            "within each block"
        ),
        "mc_variance_method": "mean(posterior_sd^2 / n_posterior_draws)",
        "negative_component_estimates": negative.to_dict(orient="records"),
        "notes": [
            f"Bar and box figures use {args.n_blocks} comparable block-level estimates per component.",
            "Remainder primarily represents model-by-dataset interaction.",
            "Conditional diagnostic rows are not pure variance components.",
            "Point figures show one Total estimate and 100 estimates for every other component in each block.",
            "Each posterior-mean MC point is posterior_sd^2 / n_posterior_draws for one paired model-dataset cell.",
            "A negative method-of-moments component is retained rather than truncated.",
            "Negative variance estimates are missing in SE-scale tables and figures because they cannot be square-rooted.",
        ],
    }
    with (output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
    (output_dir / "COMPLETE").write_text("variance decomposition complete\n")

    print(f"STEP 10c complete: {output_dir}")
    print(f"Block components: {components_path}")
    print(f"Variance summary: {variance_summary_path}")
    print(f"SE summary:       {se_summary_path}")
    print(f"Diagnostics:      {details_path}")
    print(f"Point estimates:  {point_estimates_path}")
    print(f"Variance figures: {variance_bar_path}, {variance_boxplot_path}")
    print(f"                  {variance_points_path}")
    print(f"SE figures:       {se_bar_path}, {se_boxplot_path}")
    print(f"                  {se_points_path}")
    if len(negative):
        print(
            f"Note: {len(negative)} method-of-moments component estimate(s) "
            "were negative and were retained; see config.json."
        )


if __name__ == "__main__":
    main()
