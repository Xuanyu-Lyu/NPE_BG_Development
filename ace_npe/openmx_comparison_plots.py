"""Dataset-level distribution and paired scatter plots for Steps 04 and 05."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


PARAMETERS = ("A", "C", "E")
PAIR_KEYS = ["N", "test_dataset_m", "parameter"]


def fixed_n_estimates_long(merged: pd.DataFrame) -> pd.DataFrame:
    """Adapt Step 04's wide results, retaining its common usable OpenMx fits."""
    finite_openmx = np.isfinite(
        merged[[f"{parameter}_openmx_est" for parameter in PARAMETERS]]
    ).all(axis=1)
    eligible = merged.loc[merged["openmx_converged"].eq(1) & finite_openmx]
    frames = []
    for parameter in PARAMETERS:
        for method, estimate_suffix, uncertainty_suffix in (
            ("OpenMx", "openmx_est", "openmx_se"),
            ("NPE", "npe_mean", "npe_sd"),
        ):
            frames.append(pd.DataFrame({
                "N": eligible["N_pairs"].to_numpy(),
                "test_dataset_m": eligible["condition_id"].to_numpy(),
                "parameter": parameter,
                "method": method,
                "truth": eligible[f"{parameter}_true"].to_numpy(),
                "estimate": eligible[f"{parameter}_{estimate_suffix}"].to_numpy(),
                "uncertainty": eligible[f"{parameter}_{uncertainty_suffix}"].to_numpy(),
            }))
    output = pd.concat(frames, ignore_index=True)
    output["error"] = output["estimate"] - output["truth"]
    output["absolute_error"] = output["error"].abs()
    return output


def valid_metric_values(group: pd.DataFrame, metric: str) -> np.ndarray:
    """Use finite estimates/truth, and finite nonnegative uncertainty for SD."""
    valid = np.isfinite(group["estimate"]) & np.isfinite(group["truth"])
    values = group[metric].to_numpy(dtype=float)
    valid &= np.isfinite(values)
    if metric == "uncertainty":
        valid &= values >= 0
    return values[valid.to_numpy()]


def paired_scatter_data(estimates: pd.DataFrame, method: str, metric: str) -> pd.DataFrame:
    """Match test IDs within N/parameter before jointly filtering both axes."""
    columns = PAIR_KEYS + ["estimate", "truth", metric]
    # The plotted metric is either uncertainty or signed error.
    openmx = estimates.loc[estimates["method"].eq("OpenMx"), columns]
    npe = estimates.loc[estimates["method"].eq(method), columns]
    paired = openmx.merge(
        npe, on=PAIR_KEYS, suffixes=("_openmx", "_npe"), validate="one_to_one"
    )
    valid = np.isfinite(paired[[
        "estimate_openmx", "estimate_npe", "truth_openmx", "truth_npe",
        f"{metric}_openmx", f"{metric}_npe",
    ]]).all(axis=1)
    if metric == "uncertainty":
        valid &= (paired["uncertainty_openmx"] >= 0) & (paired["uncertainty_npe"] >= 0)
    if not np.allclose(
        paired.loc[valid, "truth_openmx"], paired.loc[valid, "truth_npe"]
    ):
        raise ValueError("Paired OpenMx and NPE rows disagree on ground truth")
    return paired.loc[valid].reset_index(drop=True)


def plot_summary_with_boxplots(
    estimates: pd.DataFrame,
    methods: tuple[str, ...],
    n_values: tuple[int, ...],
    colors: dict[str, str],
    metric: str,
    title: str,
    path: Path,
) -> None:
    """Show median SD or MAE above the corresponding test-level distributions."""
    n_ranges = [
        (label, values) for label, values in (
            ("N ≤ 1,000", tuple(n for n in n_values if n <= 1_000)),
            ("N > 1,000", tuple(n for n in n_values if n > 1_000)),
        ) if values
    ]
    uncertainty = metric == "uncertainty"
    summary_label = "Median uncertainty" if uncertainty else "Mean absolute error"
    distribution_label = "Uncertainty (OpenMx SE / NPE SD)" if uncertainty else "Absolute error |estimate - truth|"
    fig, axes = plt.subplots(
        2 * len(n_ranges), len(PARAMETERS),
        figsize=(16, 7.0 * len(n_ranges)), squeeze=False,
    )
    width = 0.8 / len(methods)
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width
    for range_index, (range_label, range_values) in enumerate(n_ranges):
        base_positions = np.arange(len(range_values))
        for column, parameter in enumerate(PARAMETERS):
            summary_axis = axes[2 * range_index, column]
            distribution_axis = axes[2 * range_index + 1, column]
            for method_index, method in enumerate(methods):
                distributions = []
                summaries = []
                mean_intervals = []
                for n_value in range_values:
                    group = estimates.loc[
                        estimates["method"].eq(method)
                        & estimates["parameter"].eq(parameter)
                        & estimates["N"].eq(n_value)
                    ]
                    values = valid_metric_values(group, metric)
                    distributions.append(values)
                    summaries.append(
                        (np.median(values) if uncertainty else values.mean())
                        if len(values) else np.nan
                    )
                    mean_intervals.append(
                        1.96 * values.std(ddof=1) / np.sqrt(len(values))
                        if len(values) > 1 else 0.0
                    )
                positions = base_positions + offsets[method_index]
                color = colors[method]
                summary_axis.bar(
                    positions, summaries, width=width * 0.88, color=color,
                    yerr=None if uncertainty else np.vstack((
                        np.minimum(summaries, mean_intervals), mean_intervals,
                    )),
                    capsize=2, error_kw={"elinewidth": 0.8},
                )
                boxplot = distribution_axis.boxplot(
                    distributions, positions=positions, widths=width * 0.88,
                    patch_artist=True, manage_ticks=False,
                    showmeans=not uncertainty, showfliers=True,
                    medianprops={"color": "black", "linewidth": 1},
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": "black", "markersize": 3},
                    whiskerprops={"color": color}, capprops={"color": color},
                    flierprops={"marker": ".", "markerfacecolor": color,
                                "markeredgecolor": color, "markersize": 2, "alpha": 0.25},
                )
                for box in boxplot["boxes"]:
                    box.set_facecolor(color)
                    box.set_edgecolor(color)
                    box.set_alpha(0.75)
            for axis, label, panel_label in (
                (summary_axis, summary_label, "Median" if uncertainty else "MAE"),
                (distribution_axis, distribution_label, "Distribution"),
            ):
                axis.set_title(f"{parameter} — {range_label}: {panel_label}")
                axis.set_ylabel(label)
                axis.set_ylim(bottom=0)
                axis.set_xticks(base_positions, [f"{n:,}" for n in range_values])
                axis.set_xlabel("Twin-pair sample size N")
                axis.grid(axis="y", alpha=0.22)
                axis.set_axisbelow(True)
    handles = [Patch(facecolor=colors[method], label=method) for method in methods]
    fig.legend(handles=handles, loc="upper center", ncol=len(methods),
               bbox_to_anchor=(0.5, 0.94), fontsize=9)
    note = "Boxes: IQR; black line: median; whiskers: 1.5 × IQR; dots: outliers."
    if not uncertainty:
        note += " White diamonds: MAE; bar error bars: normal 95% CI of MAE."
    fig.suptitle(f"{title}\n{note}", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paired_scatter(
    estimates: pd.DataFrame,
    method: str,
    n_values: tuple[int, ...],
    metric: str,
    title: str,
    path: Path,
) -> None:
    """One test dataset per point, with separate N and ACE parameter panels."""
    paired = paired_scatter_data(estimates, method, metric)
    uncertainty = metric == "uncertainty"
    fig, axes = plt.subplots(
        len(n_values), len(PARAMETERS), figsize=(14, 3.2 * len(n_values)),
        squeeze=False,
    )
    for row, n_value in enumerate(n_values):
        for column, parameter in enumerate(PARAMETERS):
            axis = axes[row, column]
            group = paired.loc[paired["N"].eq(n_value) & paired["parameter"].eq(parameter)]
            x = group[f"{metric}_openmx"].to_numpy(dtype=float)
            y = group[f"{metric}_npe"].to_numpy(dtype=float)
            axis.scatter(x, y, s=9, alpha=0.35, edgecolors="none")
            if len(group):
                low = 0.0 if uncertainty else min(x.min(), y.min(), 0.0)
                high = max(x.max(), y.max(), 0.0)
                padding = max((high - low) * 0.04, 1e-6)
                limits = (0.0 if uncertainty else low - padding, high + padding)
                axis.set_xlim(limits)
                axis.set_ylim(limits)
                axis.plot(limits, limits, "--", color="black", linewidth=0.9)
                if not uncertainty:
                    axis.axhline(0, color="grey", linewidth=0.6)
                    axis.axvline(0, color="grey", linewidth=0.6)
            else:
                axis.text(0.5, 0.5, "No valid paired values", transform=axis.transAxes, ha="center")
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"{parameter} — N={n_value:,} (valid pairs={len(group):,})")
            axis.set_xlabel("OpenMx SE" if uncertainty else "OpenMx signed error (estimate - truth)")
            axis.set_ylabel(f"{method} posterior SD" if uncertainty else f"{method} signed error (estimate - truth)")
            axis.grid(alpha=0.2)
    fig.suptitle(f"{title}\nOne point per test dataset; dashed line: y = x", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.12 / len(n_values)))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
