"""STEP 09 -- compare fixed-theta uncertainty sources.

This script reuses the fully crossed STEP 08 evaluation: 100 independently
trained NPEs evaluated on the same 500 fixed-theta datasets. The datasets are
randomly divided into five blocks of 100. For every ACE parameter it reports:

* Total: one variance across 100 one-to-one model/dataset posterior means per
  block (five estimates overall).
* Model: one variance across the 100 models for every dataset (100 estimates
  per block).
* Dataset: one variance across a block's 100 datasets for every model (100
  estimates per block).
* Posterior uncertainty: the posterior variance for every paired model/dataset
  cell (100 estimates per block).

Parallel variance-scale and SE-scale figures are written. These quantities are
an uncertainty comparison, not an additive variance decomposition. No NPE is
retrained and no new posterior samples are required.
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
    "Posterior uncertainty",
]
COMPONENT_COLORS = {
    "Total": "#4c4c4c",
    "Model": "#4c78a8",
    "Dataset": "#f58518",
    "Posterior uncertainty": "#b279a2",
}
PARAMETER_COLORS = {"A": "#1f77b4", "C": "#ff7f0e", "E": "#2ca02c"}


def load_evaluation(
    ensemble_dir: Path,
    n_replicates: int,
    requested_draws: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, int]:
    """Load and validate the completed, shared-dataset STEP 08 results."""
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
                f"Missing completed STEP 08 replicate {replicate}: {result_path}"
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
                    "STEP 09 requires shared test datasets, but replicate "
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
            f"STEP 08 used {next(iter(recorded_draw_counts))} posterior draws, "
            f"but STEP 09 was given --n_posterior_draws={requested_draws}"
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


def calculate_results(
    means: dict[str, np.ndarray],
    posterior_sds: dict[str, np.ndarray],
    dataset_ids: np.ndarray,
    blocks: list[np.ndarray],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the requested block and conditional uncertainty estimates."""
    estimate_rows: list[dict] = []
    assignment_rows: list[dict] = []
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
            estimate_rows.append(
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

            for local_index, dataset_position in enumerate(positions):
                estimate_rows.append(
                    {
                        "block": block_number,
                        "parameter": parameter,
                        "component": "Model",
                        "unit_type": "dataset",
                        "model": np.nan,
                        "test_simulation": int(dataset_ids[dataset_position]),
                        "variance": float(
                            block_values[:, local_index].var(ddof=1)
                        ),
                    }
                )

            for model_index in range(n_models):
                estimate_rows.append(
                    {
                        "block": block_number,
                        "parameter": parameter,
                        "component": "Dataset",
                        "unit_type": "model",
                        "model": model_index + 1,
                        "test_simulation": np.nan,
                        "variance": float(
                            block_values[model_index, :].var(ddof=1)
                        ),
                    }
                )
                paired_local_index = int(pairing[model_index])
                paired_dataset_position = int(positions[paired_local_index])
                estimate_rows.append(
                    {
                        "block": block_number,
                        "parameter": parameter,
                        "component": "Posterior uncertainty",
                        "unit_type": "paired_model_dataset",
                        "model": model_index + 1,
                        "test_simulation": int(
                            dataset_ids[paired_dataset_position]
                        ),
                        "variance": float(block_sds[model_index, paired_local_index] ** 2),
                    }
                )

    assignments = pd.DataFrame(assignment_rows)
    estimates = pd.DataFrame(estimate_rows)
    estimates["component"] = pd.Categorical(
        estimates["component"], categories=COMPONENT_ORDER, ordered=True
    )
    estimates = estimates.sort_values(
        ["parameter", "block", "component"]
    ).reset_index(drop=True)
    estimates["standard_error"] = np.sqrt(estimates["variance"])
    return estimates, assignments


def component_summary(
    estimates: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    summary = (
        estimates.groupby(["parameter", "component"], observed=True)[value_column]
        .agg(
            n_estimates="count",
            mean="mean",
            sd="std",
            minimum="min",
            median="median",
            maximum="max",
        )
        .reset_index()
    )
    return summary


def save_bar_plot(
    summary: pd.DataFrame,
    output_path: Path,
    y_label: str,
    title_label: str,
) -> None:
    total_count = int(
        summary.loc[summary["component"] == "Total", "n_estimates"].iloc[0]
    )
    conditional_count = int(
        summary.loc[summary["component"] != "Total", "n_estimates"].max()
    )
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
        f"Fixed-theta {title_label}\n"
        f"Total: {total_count} estimates; other sources: {conditional_count}; "
        "error bars: ±1 SD"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_boxplot(
    estimates: pd.DataFrame,
    output_path: Path,
    value_column: str,
    y_label: str,
    title_label: str,
) -> None:
    n_blocks = int(estimates["block"].nunique())
    per_block = int(
        estimates.loc[estimates["component"] != "Total"]
        .groupby(["parameter", "block", "component"], observed=True)
        .size()
        .max()
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=False)
    positions = np.arange(1, len(COMPONENT_ORDER) + 1)
    rng = np.random.default_rng(20260925)
    for axis, parameter in zip(axes, ACE_PARAM_NAMES):
        selected = estimates.loc[estimates["parameter"] == parameter]
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
            is_total = component == "Total"
            axis.scatter(
                position + jitter,
                component_values,
                s=42 if is_total else 12,
                color=COMPONENT_COLORS[component],
                alpha=0.90 if is_total else 0.25,
                edgecolor="white" if is_total else "none",
                linewidth=0.4 if is_total else 0,
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
        f"Total: one estimate per block; other sources: {per_block} estimates per block"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
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
        raise FileNotFoundError(f"STEP 08 evaluation directory not found: {ensemble_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    means, posterior_sds, dataset_ids, n_draws = load_evaluation(
        ensemble_dir, args.n_replicates, args.n_posterior_draws
    )
    rng = np.random.default_rng(args.seed)
    blocks = make_blocks(dataset_ids, args.n_replicates, args.n_blocks, rng)
    estimates, assignments = calculate_results(
        means,
        posterior_sds,
        dataset_ids,
        blocks,
        rng,
    )
    variance_summary = component_summary(estimates, "variance")
    se_summary = component_summary(estimates, "standard_error")

    variance_summary_path = output_dir / "variance_component_summary.csv"
    se_summary_path = output_dir / "se_component_summary.csv"
    estimates_path = output_dir / "component_point_estimates.csv"
    assignments_path = output_dir / "block_pairing_assignments.csv"
    variance_summary.to_csv(variance_summary_path, index=False)
    se_summary.to_csv(se_summary_path, index=False)
    estimates.to_csv(estimates_path, index=False)
    assignments.to_csv(assignments_path, index=False)

    # Remove tables from the former additive-decomposition version so a rerun
    # cannot leave obsolete results beside the new uncertainty comparison.
    for obsolete_name in (
        "variance_components_by_block.csv",
        "conditional_variance_diagnostics.csv",
    ):
        (output_dir / obsolete_name).unlink(missing_ok=True)

    variance_bar_path = output_dir / "variance_components_mean_bar.png"
    variance_boxplot_path = output_dir / "variance_components_boxplot.png"
    variance_points_path = output_dir / "variance_components_block_points.png"
    save_bar_plot(
        variance_summary,
        variance_bar_path,
        "Variance",
        "uncertainty comparison on the variance scale",
    )
    save_boxplot(
        estimates,
        variance_boxplot_path,
        "variance",
        "Variance",
        "uncertainty comparison on the variance scale",
    )
    save_block_point_plot(
        estimates,
        variance_points_path,
        "variance",
        "Variance",
        "uncertainty sources on the variance scale",
    )

    se_bar_path = output_dir / "se_components_mean_bar.png"
    se_boxplot_path = output_dir / "se_components_boxplot.png"
    se_points_path = output_dir / "se_components_block_points.png"
    save_bar_plot(
        se_summary,
        se_bar_path,
        "SE = sqrt(variance)",
        "uncertainty comparison on the SE scale",
    )
    save_boxplot(
        estimates,
        se_boxplot_path,
        "standard_error",
        "SE = sqrt(variance)",
        "uncertainty comparison on the SE scale",
    )
    save_block_point_plot(
        estimates,
        se_points_path,
        "standard_error",
        "SE = sqrt(variance)",
        "uncertainty sources on the SE scale",
    )

    config = {
        "experiment": "fixed_theta_uncertainty_comparison",
        "source_ensemble_dir": str(ensemble_dir),
        "n_models": args.n_replicates,
        "n_datasets": int(len(dataset_ids)),
        "n_blocks": args.n_blocks,
        "datasets_per_block": int(len(dataset_ids) // args.n_blocks),
        "n_posterior_draws": n_draws,
        "seed": args.seed,
        "parameters": list(ACE_PARAM_NAMES),
        "components": list(COMPONENT_ORDER),
        "total_variance_method": (
            "sample variance of a random one-to-one model/dataset pairing "
            "within each block"
        ),
        "model_variance_method": (
            "sample variance across models, calculated separately for every dataset"
        ),
        "dataset_variance_method": (
            "sample variance across datasets, calculated separately for every model and block"
        ),
        "posterior_uncertainty_method": (
            "posterior_sd^2 for each paired model/dataset cell"
        ),
        "notes": [
            f"Total has {args.n_blocks} estimates, one per block.",
            f"Model, Dataset, and Posterior uncertainty each have {args.n_replicates} estimates per block.",
            "These quantities are an uncertainty comparison and are not additive variance components.",
        ],
    }
    with (output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
    (output_dir / "COMPLETE").write_text("uncertainty comparison complete\n")

    print(f"STEP 09 complete: {output_dir}")
    print(f"Point estimates:  {estimates_path}")
    print(f"Variance summary: {variance_summary_path}")
    print(f"SE summary:       {se_summary_path}")
    print(f"Variance figures: {variance_bar_path}, {variance_boxplot_path}")
    print(f"                  {variance_points_path}")
    print(f"SE figures:       {se_bar_path}, {se_boxplot_path}")
    print(f"                  {se_points_path}")


if __name__ == "__main__":
    main()
