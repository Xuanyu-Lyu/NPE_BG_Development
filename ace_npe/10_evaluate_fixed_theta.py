"""STEP 10 -- evaluate 100k-simulation fixed-N NPEs at one ACE condition.

The default experiment evaluates the existing N=50, 100, 500, and 1000 NPEs
trained from 100,000 prior-predictive simulations per N. For each N it
generates 500 independent MZ/DZ datasets at
theta=(A, C, E)=(0.4, 0.3, 0.3) and compares:

* the empirical SE of the posterior-mean estimator (the sample SD of posterior
  means across repeated datasets); and
* the root-mean-square posterior SE, ``sqrt(mean(posterior variance))``.

The mean posterior SE is retained in the output tables as a secondary
descriptive metric, but the calibration plot uses the RMS posterior SE because
sampling variance should be compared with average posterior variance.

Metrics are reported for all 500 simulations and for five non-overlapping
blocks of 100. The calibration plot uses color for N, large points for the
all-500 results, and small translucent points for the five blocks. No NPE is
trained or modified by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_model import (  # noqa: E402
    ACE_PARAM_NAMES,
    COV_FEATURE_NAMES,
    MODELS_DIR,
    RESULTS_DIR,
    load_posterior,
    resolve,
    summarize_cov,
    theoretical_covariances,
)


DEFAULT_N_PAIRS = (50, 100, 500, 1000)
DEFAULT_MODELS_DIR = "prior_comparison_100k_N20000/dirichlet"
DEFAULT_OUTPUT_DIR = "fixed_theta_comparison_100k"
DEFAULT_EXPECTED_MODEL_SIMULATIONS = 100_000
RMS_POSTERIOR_SE_COLUMN = "sqrt(mean(posterior_var))"


def validate_model(
    loaded: dict,
    n_pairs: int,
    expected_model_simulations: int | None,
) -> None:
    """Ensure the loaded run is the requested fixed-N Dirichlet model."""
    config = loaded["config"]
    if config.get("simulation_scheme") != "dirichlet":
        raise ValueError(f"N={n_pairs}: selected model is not a Dirichlet NPE")
    if config.get("fixed_n_pairs") != n_pairs:
        raise ValueError(
            f"N={n_pairs}: model was trained at fixed N="
            f"{config.get('fixed_n_pairs')}"
        )
    if config.get("n_is_model_feature") is not False:
        raise ValueError(f"N={n_pairs}: expected a fixed-N model with no N feature")
    if list(loaded["feature_cols"]) != list(COV_FEATURE_NAMES):
        raise ValueError(
            f"N={n_pairs}: model features are {loaded['feature_cols']}; "
            f"expected {list(COV_FEATURE_NAMES)}"
        )
    if list(loaded["param_names"]) != list(ACE_PARAM_NAMES):
        raise ValueError(
            f"N={n_pairs}: model parameters are {loaded['param_names']}; "
            f"expected {list(ACE_PARAM_NAMES)}"
        )
    split_columns = ("training_rows", "validation_rows", "test_rows")
    split_counts = [config.get(column) for column in split_columns]
    if expected_model_simulations is not None:
        if any(value is None for value in split_counts):
            raise ValueError(
                f"N={n_pairs}: model config does not record all of "
                f"{split_columns}, so its simulation budget cannot be verified"
            )
        observed_total = sum(int(value) for value in split_counts)
        if observed_total != expected_model_simulations:
            raise ValueError(
                f"N={n_pairs}: model used {observed_total:,} total simulations; "
                f"expected {expected_model_simulations:,}. Select the 100k model "
                "directory or change --expected_model_simulations."
            )


def simulate_features(
    theta: np.ndarray,
    n_pairs: int,
    n_simulations: int,
    seed: int,
) -> np.ndarray:
    """Simulate independent covariance-summary feature vectors."""
    mz_population, dz_population = theoretical_covariances(*theta)
    features = np.empty(
        (n_simulations, len(COV_FEATURE_NAMES)), dtype=np.float32
    )

    for simulation_index in range(n_simulations):
        # N is part of each seed so datasets are independent across sample-size
        # arms as well as across simulations and zygosity groups.
        mz_rng = np.random.default_rng(
            np.random.SeedSequence([seed, n_pairs, simulation_index + 1, 1])
        )
        dz_rng = np.random.default_rng(
            np.random.SeedSequence([seed, n_pairs, simulation_index + 1, 2])
        )
        mz_pairs = mz_rng.multivariate_normal(
            mean=np.zeros(2), cov=mz_population, size=n_pairs, check_valid="raise"
        )
        dz_pairs = dz_rng.multivariate_normal(
            mean=np.zeros(2), cov=dz_population, size=n_pairs, check_valid="raise"
        )
        mz_var, mz_cov = summarize_cov(np.cov(mz_pairs, rowvar=False, ddof=1))
        dz_var, dz_cov = summarize_cov(np.cov(dz_pairs, rowvar=False, ddof=1))
        features[simulation_index] = (mz_var, mz_cov, dz_var, dz_cov)

    return features


def evaluate_posterior(
    loaded: dict,
    raw_features: np.ndarray,
    theta: np.ndarray,
    n_pairs: int,
    n_posterior_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Return posterior means and SDs for every simulated dataset."""
    scaled_features = loaded["scaler"].transform(raw_features).astype(np.float32)
    posterior = loaded["posterior"]
    posterior_seed = int(
        np.random.SeedSequence([seed, n_pairs, 3]).generate_state(1)[0]
    )
    torch.manual_seed(posterior_seed)

    rows = []
    total = len(scaled_features)
    progress_every = max(1, total // 10)
    for simulation_index, scaled_row in enumerate(scaled_features, start=1):
        x = torch.as_tensor(scaled_row, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            samples = posterior.sample(
                (n_posterior_samples,), x=x, show_progress_bars=False
            )
        draws = samples.cpu().numpy()
        posterior_mean = draws.mean(axis=0)
        posterior_sd = draws.std(axis=0, ddof=1)

        raw_row = raw_features[simulation_index - 1]
        row = {
            "N_pairs": n_pairs,
            "simulation": simulation_index,
            **{
                feature: float(value)
                for feature, value in zip(COV_FEATURE_NAMES, raw_row)
            },
        }
        for parameter_index, parameter in enumerate(ACE_PARAM_NAMES):
            row[f"true_{parameter}"] = float(theta[parameter_index])
            row[f"{parameter}_posterior_mean"] = float(
                posterior_mean[parameter_index]
            )
            row[f"{parameter}_posterior_sd"] = float(
                posterior_sd[parameter_index]
            )
        rows.append(row)

        if simulation_index % progress_every == 0 or simulation_index == total:
            print(
                f"  N={n_pairs}: evaluated "
                f"{simulation_index}/{total} simulated datasets"
            )

    return pd.DataFrame(rows)


def summarize_results(
    results: pd.DataFrame,
    n_values: tuple[int, ...],
    group_size: int,
) -> pd.DataFrame:
    """Summarize each N over all simulations and consecutive blocks."""
    rows = []
    for n_pairs in n_values:
        at_n = (
            results.loc[results["N_pairs"] == n_pairs]
            .sort_values("simulation")
            .reset_index(drop=True)
        )
        groups: list[tuple[str, pd.DataFrame]] = [
            (f"All {len(at_n)}", at_n)
        ]
        for start in range(0, len(at_n), group_size):
            stop = min(start + group_size, len(at_n))
            groups.append((f"{start + 1}-{stop}", at_n.iloc[start:stop]))

        for group_label, group in groups:
            for parameter in ACE_PARAM_NAMES:
                posterior_sds = group[f"{parameter}_posterior_sd"]
                rows.append(
                    {
                        "N_pairs": n_pairs,
                        "group": group_label,
                        "n_simulations": len(group),
                        "parameter": parameter,
                        # Repository mc_se: SD of the posterior-mean estimator,
                        # not the Monte Carlo error SD/sqrt(R).
                        "SE(mean(theta))": group[
                            f"{parameter}_posterior_mean"
                        ].std(ddof=1),
                        "mean(posterior_SE)": posterior_sds.mean(),
                        RMS_POSTERIOR_SE_COLUMN: np.sqrt(
                            np.mean(np.square(posterior_sds))
                        ),
                    }
                )
    return pd.DataFrame(rows)


def make_wide_table(
    summary: pd.DataFrame,
    n_values: tuple[int, ...],
) -> pd.DataFrame:
    """Format the requested A/C/E metric pairs as one compact table."""
    group_order = {
        group: position
        for position, group in enumerate(summary["group"].drop_duplicates())
    }
    n_order = {n_pairs: position for position, n_pairs in enumerate(n_values)}
    wide = summary.pivot(
        index=["N_pairs", "group", "n_simulations"],
        columns="parameter",
        values=[
            "SE(mean(theta))",
            "mean(posterior_SE)",
            RMS_POSTERIOR_SE_COLUMN,
        ],
    )
    wide = wide.swaplevel(0, 1, axis=1).reindex(
        columns=ACE_PARAM_NAMES, level=0
    )
    wide.columns = [
        f"{parameter}: {metric}" for parameter, metric in wide.columns
    ]
    wide = wide.reset_index()
    wide["_n_order"] = wide["N_pairs"].map(n_order)
    wide["_group_order"] = wide["group"].map(group_order)
    return (
        wide.sort_values(["_n_order", "_group_order"])
        .drop(columns=["_n_order", "_group_order"])
        .reset_index(drop=True)
    )


def save_calibration_plot(
    summary: pd.DataFrame,
    n_values: tuple[int, ...],
    n_simulations: int,
    group_size: int,
    theta: np.ndarray,
    path: Path,
) -> None:
    """Plot RMS posterior SE against empirical SE, with block stability points."""
    cmap = plt.get_cmap("viridis")
    color_positions = np.linspace(0.12, 0.88, len(n_values))
    colors = {
        n_pairs: cmap(position)
        for n_pairs, position in zip(n_values, color_positions)
    }
    all_group = f"All {n_simulations}"

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), squeeze=False)
    for col, parameter in enumerate(ACE_PARAM_NAMES):
        axis = axes[0, col]
        at_parameter = summary.loc[summary["parameter"] == parameter]
        maximum = 0.0

        for n_pairs in n_values:
            at_n = at_parameter.loc[at_parameter["N_pairs"] == n_pairs]
            blocks = at_n.loc[at_n["group"] != all_group]
            overall = at_n.loc[at_n["group"] == all_group]
            if len(overall) != 1:
                raise ValueError(
                    f"Expected one all-{n_simulations} row for "
                    f"parameter={parameter}, N={n_pairs}"
                )
            color = colors[n_pairs]
            axis.scatter(
                blocks["SE(mean(theta))"],
                blocks[RMS_POSTERIOR_SE_COLUMN],
                s=34,
                color=color,
                alpha=0.48,
                edgecolors="none",
                zorder=2,
            )
            x_value = float(overall["SE(mean(theta))"].iloc[0])
            y_value = float(overall[RMS_POSTERIOR_SE_COLUMN].iloc[0])
            axis.scatter(
                [x_value],
                [y_value],
                s=135,
                color=color,
                edgecolor="black",
                linewidth=0.9,
                zorder=4,
            )
            axis.annotate(
                f"N={n_pairs}",
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8.5,
                color=color,
            )
            maximum = max(
                maximum,
                float(at_n["SE(mean(theta))"].max()),
                float(at_n[RMS_POSTERIOR_SE_COLUMN].max()),
            )

        limit = maximum * 1.12
        axis.plot([0.0, limit], [0.0, limit], "--", color="0.35", linewidth=1.2)
        axis.set_xlim(0.0, limit)
        axis.set_ylim(0.0, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(parameter)
        axis.set_xlabel("SE of posterior means across datasets")
        axis.set_ylabel("sqrt(mean posterior variance)")
        axis.grid(alpha=0.22)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colors[n_pairs],
            markeredgecolor="none",
            markersize=8,
            label=f"N={n_pairs}",
        )
        for n_pairs in n_values
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="0.65",
                markeredgecolor="black",
                markersize=11,
                label=f"All {n_simulations}",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="0.65",
                markeredgecolor="none",
                alpha=0.48,
                markersize=6,
                label=f"Non-overlapping blocks of {group_size}",
            ),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Fixed-theta NPE uncertainty comparison\n"
        f"True (A, C, E)=({theta[0]:g}, {theta[1]:g}, {theta[2]:g}); "
        "dashed line: sqrt(mean posterior variance) = empirical SE"
    )
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 0.92))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-N Dirichlet NPEs at one fixed ACE condition"
    )
    parser.add_argument("--models_dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--expected_model_simulations",
        type=int,
        default=DEFAULT_EXPECTED_MODEL_SIMULATIONS,
        help=(
            "Required total training-corpus size (train + validation + held-out "
            "test rows); use 0 to disable this check (default: 100000)"
        ),
    )
    parser.add_argument(
        "--n_pairs", type=int, nargs="+", default=list(DEFAULT_N_PAIRS)
    )
    parser.add_argument("--theta", type=float, nargs=3, default=[0.4, 0.3, 0.3])
    parser.add_argument("--n_simulations", type=int, default=500)
    parser.add_argument("--group_size", type=int, default=100)
    parser.add_argument("--n_posterior_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    n_values = tuple(dict.fromkeys(int(n_pairs) for n_pairs in args.n_pairs))
    if any(n_pairs < 2 for n_pairs in n_values):
        parser.error("--n_pairs values must be at least 2")
    if args.n_simulations < 2:
        parser.error("--n_simulations must be at least 2")
    if args.group_size < 2 or args.group_size > args.n_simulations:
        parser.error("--group_size must be between 2 and --n_simulations")
    if args.n_posterior_samples < 2:
        parser.error("--n_posterior_samples must be at least 2")
    if args.expected_model_simulations < 0:
        parser.error("--expected_model_simulations must be non-negative")

    theta = np.asarray(args.theta, dtype=float)
    if theta.shape != (3,) or np.any(theta <= 0):
        parser.error("--theta must contain three positive values")
    if not np.isclose(theta.sum(), 1.0):
        parser.error("The fixed-N Dirichlet NPE requires A+C+E=1")

    models_dir = resolve(args.models_dir, MODELS_DIR)
    output_dir = resolve(args.output_dir, RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_model_simulations = (
        args.expected_model_simulations
        if args.expected_model_simulations > 0
        else None
    )

    frames = []
    model_dirs = {}
    print("Loading and evaluating existing NPEs (no retraining) ...")
    for n_pairs in n_values:
        model_dir = models_dir / f"N{n_pairs}"
        loaded = load_posterior(model_dir)
        validate_model(loaded, n_pairs, expected_model_simulations)
        model_dirs[n_pairs] = str(model_dir)
        print(f"\nN={n_pairs} model: {model_dir}")
        print(
            f"Simulating {args.n_simulations} datasets at "
            f"theta=({theta[0]}, {theta[1]}, {theta[2]}) ..."
        )
        raw_features = simulate_features(
            theta=theta,
            n_pairs=n_pairs,
            n_simulations=args.n_simulations,
            seed=args.seed,
        )
        frames.append(
            evaluate_posterior(
                loaded=loaded,
                raw_features=raw_features,
                theta=theta,
                n_pairs=n_pairs,
                n_posterior_samples=args.n_posterior_samples,
                seed=args.seed,
            )
        )

    results = pd.concat(frames, ignore_index=True)
    summary = summarize_results(results, n_values, args.group_size)
    wide_summary = make_wide_table(summary, n_values)

    raw_path = output_dir / "fixed_theta_posterior_results.csv"
    summary_path = output_dir / "fixed_theta_summary_long.csv"
    table_path = output_dir / "fixed_theta_summary_table.csv"
    plot_path = output_dir / "fixed_theta_se_comparison.png"
    config_path = output_dir / "config.json"
    results.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    wide_summary.to_csv(table_path, index=False)
    save_calibration_plot(
        summary=summary,
        n_values=n_values,
        n_simulations=args.n_simulations,
        group_size=args.group_size,
        theta=theta,
        path=plot_path,
    )
    with open(config_path, "w") as handle:
        json.dump(
            {
                "model_dirs": model_dirs,
                "expected_model_simulations": expected_model_simulations,
                "n_pairs": list(n_values),
                "theta": dict(zip(ACE_PARAM_NAMES, theta.tolist())),
                "n_simulations_per_n": args.n_simulations,
                "group_size": args.group_size,
                "n_posterior_samples": args.n_posterior_samples,
                "seed": args.seed,
                "definition": {
                    "SE(mean(theta))": (
                        "sample SD (ddof=1) of posterior means across "
                        "simulated datasets"
                    ),
                    "mean(posterior_SE)": (
                        "mean posterior sample SD (ddof=1) across "
                        "simulated datasets"
                    ),
                    RMS_POSTERIOR_SE_COLUMN: (
                        "square root of the mean posterior sample variance "
                        "across simulated datasets; primary plot metric"
                    ),
                    "plot_large_points": (
                        f"metrics calculated from all {args.n_simulations} "
                        "datasets at each N"
                    ),
                    "plot_small_points": (
                        f"metrics calculated in consecutive non-overlapping "
                        f"blocks of {args.group_size} datasets"
                    ),
                },
            },
            handle,
            indent=2,
        )

    print("\nSummary:")
    print(
        wide_summary.to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        )
    )
    print(f"\nRaw results:  {raw_path}")
    print(f"Summary table: {table_path}")
    print(f"Plot:          {plot_path}")


if __name__ == "__main__":
    main()
