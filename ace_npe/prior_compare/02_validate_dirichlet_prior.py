"""PRIOR COMPARE STEP 02 -- validate a simulated Dirichlet ACE dataset.

The script checks the simplex constraint and compares empirical marginal
moments with their Dirichlet values. It also saves a ternary density plot.
N_pairs labels the fixed twin-pair sample size; the ACE prior itself does not
depend on the number of simulated twin pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from scipy.stats import beta, kstest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ace_model import ACE_PARAM_NAMES, DATA_DIR, RESULTS_DIR, resolve


SQRT3_OVER_2 = np.sqrt(3.0) / 2.0


def barycentric_to_xy(ace_proportions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map (A, C, E) simplex coordinates to an equilateral triangle."""
    values = np.asarray(ace_proportions, dtype=float)
    x = values[..., 1] + 0.5 * values[..., 2]
    y = SQRT3_OVER_2 * values[..., 2]
    return x, y


def draw_ternary_frame(axis) -> Polygon:
    """Draw the simplex boundary and 0.2-spaced ternary grid."""
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, SQRT3_OVER_2]])
    boundary = Polygon(vertices, closed=True, fill=False, edgecolor="black", linewidth=1.5)
    axis.add_patch(boundary)

    for value in np.arange(0.2, 1.0, 0.2):
        line_endpoints = (
            # A = value
            np.array([[value, 1.0 - value, 0.0], [value, 0.0, 1.0 - value]]),
            # C = value
            np.array([[1.0 - value, value, 0.0], [0.0, value, 1.0 - value]]),
            # E = value
            np.array([[1.0 - value, 0.0, value], [0.0, 1.0 - value, value]]),
        )
        for endpoints in line_endpoints:
            x, y = barycentric_to_xy(endpoints)
            axis.plot(x, y, color="white", alpha=0.45, linewidth=0.7, zorder=3)

    axis.text(-0.025, -0.035, "A = 1", ha="right", va="top", fontsize=11)
    axis.text(1.0, -0.035, "C = 1", ha="center", va="top", fontsize=11)
    axis.text(0.5, SQRT3_OVER_2 + 0.035, "E = 1", ha="center", va="bottom", fontsize=11)
    axis.set_xlim(-0.08, 1.08)
    axis.set_ylim(-0.07, SQRT3_OVER_2 + 0.08)
    axis.set_aspect("equal")
    axis.axis("off")
    return boundary


def read_manifest_defaults(data_path: Path):
    manifest_path = data_path.parent / "simulation_manifest.json"
    if not manifest_path.exists():
        return None, None
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    return manifest.get("dirichlet_alpha"), manifest.get("total_variance_fixed_sum")


def theoretical_moments(alpha: np.ndarray, total_variance: float):
    alpha0 = alpha.sum()
    mean = total_variance * alpha / alpha0
    covariance = np.empty((3, 3), dtype=float)
    denominator = alpha0**2 * (alpha0 + 1.0)
    for i in range(3):
        for j in range(3):
            if i == j:
                covariance[i, j] = (
                    total_variance**2
                    * alpha[i]
                    * (alpha0 - alpha[i])
                    / denominator
                )
            else:
                covariance[i, j] = (
                    -total_variance**2 * alpha[i] * alpha[j] / denominator
                )
    return mean, covariance


def validation_table(
    ace: np.ndarray,
    alpha: np.ndarray,
    total_variance: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    expected_mean, expected_covariance = theoretical_moments(alpha, total_variance)
    observed_mean = ace.mean(axis=0)
    observed_sd = ace.std(axis=0, ddof=1)
    expected_sd = np.sqrt(np.diag(expected_covariance))
    proportions = ace / total_variance
    alpha0 = alpha.sum()

    rows = []
    for i, parameter in enumerate(ACE_PARAM_NAMES):
        mean_mc_se = expected_sd[i] / np.sqrt(len(ace))
        ks = kstest(proportions[:, i], beta(alpha[i], alpha0 - alpha[i]).cdf)
        rows.append(
            {
                "parameter": parameter,
                "observed_mean": observed_mean[i],
                "expected_mean": expected_mean[i],
                "mean_error": observed_mean[i] - expected_mean[i],
                "mean_error_mc_se_units": (
                    (observed_mean[i] - expected_mean[i]) / mean_mc_se
                ),
                "observed_sd": observed_sd[i],
                "expected_sd": expected_sd[i],
                "ks_statistic_vs_beta_marginal": ks.statistic,
                "ks_p_value": ks.pvalue,
            }
        )
    return pd.DataFrame(rows), expected_covariance, np.cov(ace, rowvar=False, ddof=1)


def save_plot(
    ace: np.ndarray,
    alpha: np.ndarray,
    total_variance: float,
    max_sum_error: float,
    n_pairs: int | None,
    gridsize: int,
    output_path: Path,
):
    proportions = ace / total_variance
    x, y = barycentric_to_xy(proportions)

    fig, ternary_axis = plt.subplots(figsize=(7.5, 7.0))
    hexagons = ternary_axis.hexbin(
        x,
        y,
        gridsize=gridsize,
        mincnt=1,
        cmap="viridis",
        linewidths=0.15,
        extent=(0.0, 1.0, 0.0, SQRT3_OVER_2),
    )
    boundary = draw_ternary_frame(ternary_axis)
    hexagons.set_clip_path(boundary)
    colorbar = fig.colorbar(hexagons, ax=ternary_axis, fraction=0.046, pad=0.02)
    colorbar.set_label("Observations per hexagonal bin")
    ternary_axis.set_title("Empirical ternary density", pad=12)

    alpha_text = ", ".join(f"{value:g}" for value in alpha)
    n_pairs_text = f"twin-pair N={n_pairs:,}, " if n_pairs is not None else ""
    fig.suptitle(
        f"Dirichlet prior validation: alpha=({alpha_text}), V={total_variance:g}, "
        f"{n_pairs_text}prior draws={len(ace):,}\n"
        f"max |A+C+E-V| = {max_sum_error:.2e}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Validate simulated Dirichlet ACE values")
    parser.add_argument(
        "--data",
        default="prior_comparison/ace_dirichlet_N100.csv",
        help="CSV path, absolute or relative to ace_npe/data/",
    )
    parser.add_argument(
        "--output_dir",
        default="prior_comparison",
        help="Output directory, absolute or relative to ace_npe/results/",
    )
    parser.add_argument("--dirichlet_alpha", type=float, nargs=3, default=None)
    parser.add_argument("--total_variance", type=float, default=None)
    parser.add_argument("--gridsize", type=int, default=32)
    parser.add_argument("--sum_tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    data_path = resolve(args.data, DATA_DIR)
    output_dir = resolve(args.output_dir, RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_alpha, manifest_total = read_manifest_defaults(data_path)
    alpha = np.asarray(
        args.dirichlet_alpha
        if args.dirichlet_alpha is not None
        else (manifest_alpha if manifest_alpha is not None else [1.0, 1.0, 1.0]),
        dtype=float,
    )
    total_variance = float(
        args.total_variance
        if args.total_variance is not None
        else (manifest_total if manifest_total is not None else 1.0)
    )
    if alpha.shape != (3,) or np.any(alpha <= 0):
        parser.error("--dirichlet_alpha must contain three positive values")
    if total_variance <= 0:
        parser.error("--total_variance must be positive")
    if args.gridsize < 5:
        parser.error("--gridsize must be at least 5")

    df = pd.read_csv(data_path)
    missing = [parameter for parameter in ACE_PARAM_NAMES if parameter not in df]
    if missing:
        raise ValueError(f"Dataset is missing ACE columns: {missing}")
    if "simulation_scheme" in df and not (df["simulation_scheme"] == "dirichlet").all():
        raise ValueError("The input contains rows not labeled as the Dirichlet scheme")

    ace = df[ACE_PARAM_NAMES].to_numpy(dtype=float)
    if not np.isfinite(ace).all():
        raise ValueError("ACE values contain NaN or infinite values")
    sums = ace.sum(axis=1)
    max_sum_error = float(np.max(np.abs(sums - total_variance)))
    minimum_component = float(ace.min())
    simplex_pass = max_sum_error <= args.sum_tolerance and minimum_component > 0.0

    metrics, expected_covariance, observed_covariance = validation_table(
        ace, alpha, total_variance
    )
    n_label = "combined"
    n_pairs = None
    if "N_pairs" in df and df["N_pairs"].nunique() == 1:
        n_pairs = int(df["N_pairs"].iloc[0])
        n_label = f"N{n_pairs}"
    plot_path = output_dir / f"dirichlet_prior_validation_{n_label}.png"
    table_path = output_dir / f"dirichlet_prior_validation_{n_label}.csv"
    save_plot(
        ace,
        alpha,
        total_variance,
        max_sum_error,
        n_pairs,
        args.gridsize,
        plot_path,
    )
    metrics.to_csv(table_path, index=False)

    print("=" * 72)
    print("DIRICHLET PRIOR VALIDATION")
    print("=" * 72)
    print(f"Data:                 {data_path}")
    print(f"Rows:                 {len(df):,}")
    print(f"Alpha:                {alpha.tolist()}")
    print(f"Total variance V:     {total_variance:g}")
    print(f"Minimum component:    {minimum_component:.6g}")
    print(f"Maximum sum error:    {max_sum_error:.3e}")
    print(f"Simplex check:        {'PASS' if simplex_pass else 'FAIL'}")
    print("\nMarginal checks:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("\nExpected covariance:")
    print(expected_covariance)
    print("Observed covariance:")
    print(observed_covariance)
    print(f"\nSaved plot:           {plot_path}")
    print(f"Saved metrics:        {table_path}")
    if not simplex_pass:
        raise ValueError("Dirichlet structural validation failed")


if __name__ == "__main__":
    main()
