"""STEP 09 -- compare N=20,000 NPEs trained with 20k, 50k, and 100k simulations.

The script reuses the paired test data and OpenMx estimates from STEP 08 but
writes every output to a separate directory. Existing models and STEP 08
results are never modified.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_model import ACE_PARAM_NAMES, MODELS_DIR, RESULTS_DIR, resolve


SCRIPT_DIR = Path(__file__).resolve().parent
STEP_08 = runpy.run_path(
    str(SCRIPT_DIR / "08_compare_fixed_n_dirichlet_openmx.py")
)
TRAINING_BUDGETS = (20000, 50000, 100000)
EXTRA_TRAINING_BUDGETS = TRAINING_BUDGETS[1:]
COMPARISON_METHODS = ("OpenMx",) + tuple(
    f"NPE ({budget // 1000}k training)" for budget in TRAINING_BUDGETS
)
COMPARISON_COLORS = {
    "OpenMx": "tab:blue",
    "NPE (20k training)": "tab:orange",
    "NPE (50k training)": "tab:green",
    "NPE (100k training)": "tab:purple",
}


def training_method(budget: int) -> str:
    return f"NPE ({budget // 1000}k training)"


def short_method_labels(methods: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        "OpenMx"
        if method == "OpenMx"
        else method.replace("NPE (", "NPE\n").replace(" training)", " train")
        for method in methods
    )


def build_training_size_comparison(
    new_metrics: dict[int, pd.DataFrame],
    source_dir: Path,
    n_pairs: int,
) -> pd.DataFrame:
    """Combine OpenMx and all NPE training sizes on common paired rows."""
    source_metrics = pd.read_csv(source_dir / "paired_metrics.csv")
    if "mean_uncertainty" not in source_metrics.columns:
        source_merged = pd.read_csv(source_dir / "paired_estimates.csv")
        source_metrics, _ = STEP_08["metric_tables"](
            source_merged, (n_pairs,)
        )
    baseline = source_metrics.loc[
        (source_metrics["N_pairs"] == n_pairs)
        & (source_metrics["method"] == "NPE")
    ].copy()
    baseline["comparison_method"] = training_method(20000)
    baseline["training_simulations"] = 20000

    first_metrics = new_metrics[EXTRA_TRAINING_BUDGETS[0]]
    openmx = first_metrics.loc[first_metrics["method"] == "OpenMx"].copy()
    openmx["comparison_method"] = "OpenMx"
    openmx["training_simulations"] = pd.NA
    frames = [openmx, baseline]
    for budget in EXTRA_TRAINING_BUDGETS:
        enlarged = new_metrics[budget].loc[
            new_metrics[budget]["method"] == "NPE"
        ].copy()
        enlarged["comparison_method"] = training_method(budget)
        enlarged["training_simulations"] = budget
        frames.append(enlarged)
    return pd.concat(frames, ignore_index=True)


def save_training_size_metric_plot(
    comparison: pd.DataFrame,
    metric: str,
    ylabel: str,
    title_metric: str,
    path: Path,
    ci_columns: tuple[str, str] | None = None,
) -> None:
    """Plot one metric for OpenMx and all NPE training budgets."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), squeeze=False)
    x = np.arange(len(COMPARISON_METHODS))
    test_set_size = int(comparison["n_total"].max())
    short_labels = short_method_labels(COMPARISON_METHODS)

    for parameter_index, parameter in enumerate(ACE_PARAM_NAMES):
        axis = axes[0, parameter_index]
        subset = (
            comparison.loc[comparison["parameter"] == parameter]
            .set_index("comparison_method")
            .reindex(COMPARISON_METHODS)
        )
        values = subset[metric].to_numpy(dtype=float)
        yerr = None
        if ci_columns is not None:
            lower = subset[ci_columns[0]].to_numpy(dtype=float)
            upper = subset[ci_columns[1]].to_numpy(dtype=float)
            yerr = np.vstack((values - lower, upper - values))
        bars = axis.bar(
            x,
            values,
            yerr=yerr,
            capsize=4 if yerr is not None else 0,
            color=[COMPARISON_COLORS[method] for method in COMPARISON_METHODS],
            width=0.68,
            alpha=0.88,
        )
        axis.bar_label(bars, labels=[f"{value:.5f}" for value in values], padding=3)
        if metric == "bias":
            axis.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
        else:
            axis.set_ylim(bottom=0.0)
        axis.set_title(parameter)
        axis.set_xticks(x, short_labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"N={int(comparison['N_pairs'].iloc[0]):,} {title_metric} comparison "
        f"(test set size = {test_set_size:,})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_training_size_comparison_plots(
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    save_training_size_metric_plot(
        comparison,
        "bias",
        "Bias (estimate - truth)",
        "bias",
        output_dir / "paired_bias_N20000.png",
        ci_columns=("bias_ci_lo", "bias_ci_hi"),
    )
    save_training_size_metric_plot(
        comparison,
        "mae",
        "Mean absolute error",
        "mean absolute error",
        output_dir / "paired_mae_N20000.png",
        ci_columns=("mae_ci_lo", "mae_ci_hi"),
    )
    save_training_size_metric_plot(
        comparison,
        "rmse",
        "RMSE",
        "RMSE",
        output_dir / "paired_rmse_N20000.png",
    )
    save_training_size_metric_plot(
        comparison,
        "mean_uncertainty",
        "Mean reported uncertainty",
        "uncertainty (OpenMx SE versus NPE posterior SD)",
        output_dir / "paired_mean_uncertainty_N20000.png",
        ci_columns=("uncertainty_mean_ci_lo", "uncertainty_mean_ci_hi"),
    )
    save_training_size_coverage_plot(
        comparison,
        output_dir / "paired_coverage_N20000.png",
    )


def save_training_size_coverage_plot(
    comparison: pd.DataFrame,
    path: Path,
) -> None:
    """Plot empirical 95% coverage for OpenMx and all NPE training sizes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), squeeze=False)
    x = np.arange(len(COMPARISON_METHODS))
    test_set_size = int(comparison["n_total"].max())
    short_labels = short_method_labels(COMPARISON_METHODS)

    for parameter_index, parameter in enumerate(ACE_PARAM_NAMES):
        axis = axes[0, parameter_index]
        subset = (
            comparison.loc[comparison["parameter"] == parameter]
            .set_index("comparison_method")
            .reindex(COMPARISON_METHODS)
        )
        values = subset["coverage_95"].to_numpy(dtype=float)
        lower = subset["coverage_95_ci_lo"].to_numpy(dtype=float)
        upper = subset["coverage_95_ci_hi"].to_numpy(dtype=float)
        colors = [COMPARISON_COLORS[method] for method in COMPARISON_METHODS]
        axis.axhline(
            0.95,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Nominal 95%",
        )
        for position, value, low, high, color in zip(
            x, values, lower, upper, colors
        ):
            axis.errorbar(
                position,
                value,
                yerr=[[value - low], [high - value]],
                fmt="o",
                markersize=8,
                capsize=4,
                color=color,
            )
            axis.annotate(
                f"{value:.3f}",
                xy=(position, value),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=9,
            )
        axis.set_ylim(max(0.0, float(lower.min()) - 0.03), 1.005)
        axis.set_title(parameter)
        axis.set_xticks(x, short_labels)
        axis.set_ylabel("Empirical 95% coverage")
        axis.grid(axis="y", alpha=0.25)

    axes[0, 0].legend()
    fig.suptitle(
        f"N={int(comparison['N_pairs'].iloc[0]):,} 95% interval coverage "
        f"(test set size = {test_set_size:,})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_training_size_shape_comparison(
    source_dir: Path,
    new_npe: dict[int, pd.DataFrame],
    n_pairs: int,
) -> pd.DataFrame:
    """Combine posterior-shape summaries for all NPE training budgets."""
    baseline_npe = pd.read_csv(source_dir / "npe_posterior_summaries.csv")
    baseline_npe = baseline_npe.loc[baseline_npe["N_pairs"] == n_pairs]
    baseline = STEP_08["posterior_shape_summary"](
        baseline_npe, (n_pairs,)
    )
    baseline["comparison_method"] = training_method(20000)
    baseline["training_simulations"] = 20000

    frames = [baseline]
    for budget in EXTRA_TRAINING_BUDGETS:
        enlarged = STEP_08["posterior_shape_summary"](
            new_npe[budget], (n_pairs,)
        )
        enlarged["comparison_method"] = training_method(budget)
        enlarged["training_simulations"] = budget
        frames.append(enlarged)
    return pd.concat(frames, ignore_index=True)


def build_training_size_shape_samples(
    source_dir: Path,
    new_npe: dict[int, pd.DataFrame],
    n_pairs: int,
) -> pd.DataFrame:
    """Combine saved per-dataset posterior summaries for all NPEs."""
    baseline = pd.read_csv(source_dir / "npe_posterior_summaries.csv")
    baseline = baseline.loc[baseline["N_pairs"] == n_pairs].copy()
    baseline["comparison_method"] = training_method(20000)
    frames = [baseline]
    for budget in EXTRA_TRAINING_BUDGETS:
        enlarged = new_npe[budget].copy()
        enlarged["comparison_method"] = training_method(budget)
        frames.append(enlarged)
    return pd.concat(frames, ignore_index=True)


def save_training_size_shape_plot(
    samples: pd.DataFrame,
    statistic: str,
    path: Path,
    n_draws: int,
) -> None:
    """Compare posterior-shape distributions for all NPE models."""
    methods = tuple(training_method(budget) for budget in TRAINING_BUDGETS)
    colors = tuple(COMPARISON_COLORS[method] for method in methods)
    short_labels = short_method_labels(methods)
    labels = {
        "skewness": ("Posterior skewness", "posterior skewness", 0.0),
        "kurtosis": (
            "Posterior kurtosis",
            "posterior kurtosis",
            3.0,
        ),
    }
    ylabel, title, normal_value = labels[statistic]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), squeeze=False)
    x = np.arange(len(methods))
    n_test_sets = int(samples.groupby("comparison_method").size().max())

    for parameter_index, parameter in enumerate(ACE_PARAM_NAMES):
        axis = axes[0, parameter_index]
        source_statistic = (
            "excess_kurtosis" if statistic == "kurtosis" else statistic
        )
        distributions = []
        for method in methods:
            values = samples.loc[
                samples["comparison_method"] == method,
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
            positions=x,
            widths=0.5,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.4},
            flierprops={
                "marker": ".",
                "markersize": 2,
                "markerfacecolor": "grey",
                "markeredgecolor": "grey",
                "alpha": 0.2,
            },
        )
        for box, color in zip(boxes["boxes"], colors):
            box.set_facecolor(color)
            box.set_edgecolor(color)
            box.set_alpha(0.35)
        for item_name in ("whiskers", "caps"):
            for item_index, item in enumerate(boxes[item_name]):
                item.set_color(colors[item_index // 2])
        if statistic == "kurtosis":
            axis.set_yscale("log")
        axis.set_title(parameter)
        axis.set_xticks(x, short_labels)
        axis.set_xlim(-0.3, len(methods) - 0.7)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)

    axes[0, 0].legend()
    if statistic == "kurtosis":
        figure_title = f"NPE {title} ({n_draws:,} posterior draws)"
    else:
        figure_title = (
            f"N={int(samples['N_pairs'].iloc[0]):,} NPE {title} "
            f"(box plots across {n_test_sets:,} test datasets)"
        )
    fig.suptitle(figure_title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 20k-, 50k-, and 100k-training N=20,000 Dirichlet NPEs"
    )
    parser.add_argument("--n_pairs", type=int, default=20000)
    parser.add_argument("--n_conditions", type=int, default=1000)
    parser.add_argument("--n_posterior_samples", type=int, default=1023)
    parser.add_argument("--sbc_bins", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--source_comparison_dir", default="dirichlet_openmx_comparison"
    )
    parser.add_argument(
        "--models_50k_dir",
        "--models_dir",
        dest="models_50k_dir",
        default="prior_comparison_50k_N20000",
    )
    parser.add_argument(
        "--models_100k_dir", default="prior_comparison_100k_N20000"
    )
    parser.add_argument(
        "--output_dir", default="dirichlet_N20000_training_size_comparison"
    )
    args = parser.parse_args()

    source_dir = resolve(args.source_comparison_dir, RESULTS_DIR)
    model_dirs = {
        50000: resolve(args.models_50k_dir, MODELS_DIR),
        100000: resolve(args.models_100k_dir, MODELS_DIR),
    }
    output_dir = resolve(args.output_dir, RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    paired = pd.read_csv(source_dir / "paired_test_data.csv")
    paired = paired.loc[paired["N_pairs"] == args.n_pairs].copy()
    openmx = pd.read_csv(source_dir / "openmx_estimates.csv")
    openmx = openmx.loc[openmx["N_pairs"] == args.n_pairs].copy()
    if len(paired) != args.n_conditions or len(openmx) != args.n_conditions:
        raise ValueError(
            f"Expected {args.n_conditions} source rows at N={args.n_pairs}; "
            f"found paired={len(paired)}, OpenMx={len(openmx)}"
        )

    n_values = (args.n_pairs,)
    npe_by_budget = {}
    merged_frames = []
    metric_frames = []
    difference_frames = []
    metrics_by_budget = {}
    for budget in EXTRA_TRAINING_BUDGETS:
        print(f"Evaluating NPE trained with {budget:,} simulations ...")
        npe = STEP_08["evaluate_npe"](
            paired,
            model_dirs[budget],
            n_values,
            args.n_posterior_samples,
            args.seed,
        )
        merged = STEP_08["merge_results"](paired, npe, openmx)
        metrics, differences = STEP_08["metric_tables"](merged, n_values)
        npe["training_simulations"] = budget
        merged["training_simulations"] = budget
        metrics["training_simulations"] = budget
        differences["training_simulations"] = budget
        npe_by_budget[budget] = npe
        metrics_by_budget[budget] = metrics
        merged_frames.append(merged)
        metric_frames.append(metrics)
        difference_frames.append(differences)

    npe_all = pd.concat(npe_by_budget.values(), ignore_index=True)
    merged_all = pd.concat(merged_frames, ignore_index=True)
    metrics_all = pd.concat(metric_frames, ignore_index=True)
    differences_all = pd.concat(difference_frames, ignore_index=True)

    paired.to_csv(output_dir / "paired_test_data.csv", index=False)
    openmx.to_csv(output_dir / "openmx_estimates.csv", index=False)
    npe_all.to_csv(output_dir / "npe_posterior_summaries.csv", index=False)
    merged_all.to_csv(output_dir / "paired_estimates.csv", index=False)
    metrics_all.to_csv(output_dir / "paired_metrics.csv", index=False)
    differences_all.to_csv(
        output_dir / "paired_estimator_differences.csv", index=False
    )

    comparison_metrics = build_training_size_comparison(
        metrics_by_budget, source_dir, args.n_pairs
    )
    comparison_metrics.loc[
        comparison_metrics["method"] == "NPE"
    ].to_csv(
        output_dir / "npe_training_size_comparison_metrics.csv", index=False
    )
    comparison_metrics.to_csv(
        output_dir / "training_size_comparison_metrics.csv", index=False
    )

    shape_comparison = build_training_size_shape_comparison(
        source_dir, npe_by_budget, args.n_pairs
    )
    shape_comparison.to_csv(
        output_dir / "npe_posterior_shape_training_comparison.csv", index=False
    )
    shape_samples = build_training_size_shape_samples(
        source_dir, npe_by_budget, args.n_pairs
    )

    save_training_size_comparison_plots(comparison_metrics, output_dir)
    save_training_size_shape_plot(
        shape_samples,
        "skewness",
        output_dir / "npe_posterior_skewness_N20000.png",
        args.n_posterior_samples,
    )
    save_training_size_shape_plot(
        shape_samples,
        "kurtosis",
        output_dir / "npe_posterior_kurtosis_N20000.png",
        args.n_posterior_samples,
    )
    for budget, npe in npe_by_budget.items():
        sbc_dir = output_dir / f"sbc_{budget // 1000}k_training"
        sbc_dir.mkdir(parents=True, exist_ok=True)
        STEP_08["save_sbc_rank_plots"](
            npe,
            n_values,
            args.n_posterior_samples,
            sbc_dir,
            n_bins=args.sbc_bins,
        )
        STEP_08["save_sbc_ecdf_plots"](
            npe,
            n_values,
            args.n_posterior_samples,
            sbc_dir,
        )

    with open(output_dir / "config.json", "w") as handle:
        json.dump(
            {
                "experiment": "N20000 Dirichlet NPE training-size comparison",
                "training_simulations": list(TRAINING_BUDGETS),
                "n_pairs": args.n_pairs,
                "n_conditions": args.n_conditions,
                "n_posterior_samples": args.n_posterior_samples,
                "sbc_bins": args.sbc_bins,
                "seed": args.seed,
                "model_dirs": {
                    str(budget): str(path) for budget, path in model_dirs.items()
                },
                "source_comparison_dir": str(source_dir),
                "source_test_data_reused": True,
                "source_openmx_results_reused": True,
                "mean_uncertainty_plot": "OpenMx standard error versus NPE posterior standard deviation",
                "coverage_interval": "95% Wilson score interval across test datasets",
                "posterior_shape_summary": "box plots across NPE test datasets; Pearson kurtosis",
            },
            handle,
            indent=2,
        )

    print(f"Saved training-size comparison -> {output_dir}")
    print(comparison_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
