"""STEP 03b -- compare matched train/test ACE simulation schemes.

At every fixed sample size, each NPE is evaluated only on fresh observations
from its own training distribution.  This gives three directly labeled
within-scheme comparisons without cross-prior cells.
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
from scipy.stats import kurtosis, skew
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_model import ACE_PARAM_NAMES, MODELS_DIR, RESULTS_DIR, load_posterior, resolve
from ace_prior_comparison import DEFAULT_N_PAIRS, SCHEMES, SCHEME_LABELS, generate_dataset, model_dir_for


def evaluate_model(loaded: dict, test_df: pd.DataFrame, n_draws: int):
    posterior = loaded["posterior"]
    scaler = loaded["scaler"]
    feature_cols = loaded["feature_cols"]
    missing = [column for column in feature_cols if column not in test_df]
    if missing:
        raise ValueError(f"OOS data are missing model features {missing}")

    x_scaled = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))
    truth = test_df[ACE_PARAM_NAMES].to_numpy(dtype=np.float32)
    means, stds, minima, maxima = [], [], [], []
    medians, lows, highs, skewnesses, kurtoses = [], [], [], [], []
    true_ranks, true_percentiles = [], []
    for observation, row in enumerate(x_scaled):
        x_obs = torch.as_tensor(row, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            samples = posterior.sample((n_draws,), x=x_obs, show_progress_bars=False)
        draws = samples.cpu().numpy()
        means.append(draws.mean(axis=0))
        stds.append(draws.std(axis=0, ddof=1))
        minima.append(draws.min(axis=0))
        maxima.append(draws.max(axis=0))
        medians.append(np.median(draws, axis=0))
        lows.append(np.percentile(draws, 2.5, axis=0))
        highs.append(np.percentile(draws, 97.5, axis=0))
        skewnesses.append(skew(draws, axis=0, bias=False))
        kurtoses.append(kurtosis(draws, axis=0, fisher=True, bias=False))
        rank = np.sum(draws < truth[observation], axis=0)
        true_ranks.append(rank)
        true_percentiles.append(rank / n_draws)
    return {
        "truth": truth,
        "mean": np.asarray(means),
        "sd": np.asarray(stds),
        "min": np.asarray(minima),
        "max": np.asarray(maxima),
        "median": np.asarray(medians),
        "ci_lo": np.asarray(lows),
        "ci_hi": np.asarray(highs),
        "skewness": np.asarray(skewnesses),
        "excess_kurtosis": np.asarray(kurtoses),
        "true_rank": np.asarray(true_ranks, dtype=int),
        "true_percentile": np.asarray(true_percentiles),
    }


def metric_rows(evaluation: dict, scheme: str, n_pairs: int):
    truth = evaluation["truth"]
    pred = evaluation["mean"]
    rows = []
    for i, parameter in enumerate(ACE_PARAM_NAMES):
        error = pred[:, i] - truth[:, i]
        bias = float(error.mean())
        bias_se = float(error.std(ddof=1) / np.sqrt(len(error)))
        rmse = float(np.sqrt(mean_squared_error(truth[:, i], pred[:, i])))
        rows.append(
            {
                "scheme": scheme,
                "N_pairs": n_pairs,
                "parameter": parameter,
                "bias": bias,
                "bias_se": bias_se,
                "bias_ci_lo": bias - 1.96 * bias_se,
                "bias_ci_hi": bias + 1.96 * bias_se,
                "rmse": rmse,
                "coverage_95": float(
                    np.mean(
                        (truth[:, i] >= evaluation["ci_lo"][:, i])
                        & (truth[:, i] <= evaluation["ci_hi"][:, i])
                    )
                ),
            }
        )
    return rows


def aggregate_row(evaluation: dict, scheme: str, n_pairs: int):
    truth = evaluation["truth"]
    pred = evaluation["mean"]
    sum_error = pred.sum(axis=1) - truth.sum(axis=1)
    coverage = (truth >= evaluation["ci_lo"]) & (truth <= evaluation["ci_hi"])
    return {
        "scheme": scheme,
        "N_pairs": n_pairs,
        "mean_coverage_95": float(coverage.mean()),
        "sum_constraint_rmse": float(np.sqrt(np.mean(sum_error**2))),
        "mean_predicted_sum": float(pred.sum(axis=1).mean()),
        "mean_true_sum": float(truth.sum(axis=1).mean()),
    }


def prediction_frame(
    evaluation: dict,
    scheme: str,
    n_pairs: int,
    test_df: pd.DataFrame,
):
    output = pd.DataFrame(
        {
            "scheme": scheme,
            "observation": np.arange(len(test_df)),
            "N_pairs": n_pairs,
        }
    )
    for i, parameter in enumerate(ACE_PARAM_NAMES):
        output[f"{parameter}_true"] = evaluation["truth"][:, i]
        output[f"{parameter}_pred"] = evaluation["mean"][:, i]
        output[f"{parameter}_sd"] = evaluation["sd"][:, i]
        output[f"{parameter}_min"] = evaluation["min"][:, i]
        output[f"{parameter}_max"] = evaluation["max"][:, i]
        output[f"{parameter}_median"] = evaluation["median"][:, i]
        output[f"{parameter}_ci_lo"] = evaluation["ci_lo"][:, i]
        output[f"{parameter}_ci_hi"] = evaluation["ci_hi"][:, i]
        output[f"{parameter}_skewness"] = evaluation["skewness"][:, i]
        output[f"{parameter}_excess_kurtosis"] = evaluation[
            "excess_kurtosis"
        ][:, i]
        output[f"{parameter}_true_rank"] = evaluation["true_rank"][:, i]
        output[f"{parameter}_true_percentile"] = evaluation[
            "true_percentile"
        ][:, i]
    output["sum_true"] = evaluation["truth"].sum(axis=1)
    output["sum_pred"] = evaluation["mean"].sum(axis=1)
    return output


def save_matched_plot(metrics: pd.DataFrame, schemes, n_values, output_dir: Path):
    """Plot each metric by N without pooling A, C, and E."""
    measures = (
        ("rmse", "RMSE", None),
        ("coverage_95", "95% coverage", 0.95),
    )
    fig, axes = plt.subplots(
        len(measures), len(ACE_PARAM_NAMES), figsize=(15, 7.5), squeeze=False
    )
    for row, (column, ylabel, reference) in enumerate(measures):
        for col, parameter in enumerate(ACE_PARAM_NAMES):
            axis = axes[row, col]
            for scheme in schemes:
                subset = metrics.loc[
                    (metrics["scheme"] == scheme)
                    & (metrics["parameter"] == parameter)
                ].set_index("N_pairs")
                values = subset.reindex(n_values)[column].to_numpy()
                axis.plot(n_values, values, marker="o", label=SCHEME_LABELS[scheme])
            axis.set_xscale("log")
            axis.set_xticks(n_values, [str(n) for n in n_values])
            axis.set_xlabel("Fixed N pairs")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            if row == 0:
                axis.set_title(parameter)
            if reference is not None:
                axis.axhline(
                    reference, color="red", linestyle="--", linewidth=1.25
                )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Matched train/test performance by parameter and sample size")
    fig.tight_layout()
    fig.savefig(
        output_dir / "matched_performance_by_n.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def save_parameter_plot(
    metrics: pd.DataFrame, schemes, n_pairs: int, output_dir: Path
):
    """Compare A/C/E metrics across the three matched schemes at one N."""
    at_n = metrics.loc[metrics["N_pairs"] == n_pairs]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    width = 0.8 / len(schemes)
    positions = np.arange(len(ACE_PARAM_NAMES))
    measures = (
        ("rmse", "RMSE", None),
        ("coverage_95", "95% coverage", 0.95),
    )
    for offset, scheme in enumerate(schemes):
        subset = at_n.loc[at_n["scheme"] == scheme].set_index("parameter")
        x = positions + (offset - (len(schemes) - 1) / 2) * width
        for axis, (column, _, _) in zip(axes, measures):
            values = subset.reindex(ACE_PARAM_NAMES)[column].to_numpy()
            axis.bar(x, values, width, label=SCHEME_LABELS[scheme])
    for axis, (_, title, reference) in zip(axes, measures):
        axis.set_xticks(positions, ACE_PARAM_NAMES)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if reference is not None:
            axis.axhline(reference, color="red", linestyle="--", linewidth=1.5)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Matched train/test performance, fixed N={n_pairs}")
    fig.tight_layout()
    fig.savefig(
        output_dir / f"matched_comparison_N{n_pairs}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_bias_plot(metrics: pd.DataFrame, schemes, n_values, output_dir: Path):
    """Plot mean signed error and its across-test-observation 95% CI."""
    colors = {
        "independent_uniform": "tab:blue",
        "normalized_uniform": "tab:orange",
        "dirichlet": "tab:green",
    }
    fig, axes = plt.subplots(
        len(schemes), len(ACE_PARAM_NAMES), figsize=(15, 10), squeeze=False
    )
    for row, scheme in enumerate(schemes):
        for col, parameter in enumerate(ACE_PARAM_NAMES):
            axis = axes[row, col]
            subset = metrics.loc[
                (metrics["scheme"] == scheme)
                & (metrics["parameter"] == parameter)
            ].set_index("N_pairs").reindex(n_values)
            bias = subset["bias"].to_numpy()
            ci_lo = subset["bias_ci_lo"].to_numpy()
            ci_hi = subset["bias_ci_hi"].to_numpy()
            color = colors[scheme]
            axis.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
            axis.fill_between(
                n_values,
                ci_lo,
                ci_hi,
                color=color,
                alpha=0.18,
                label="95% CI",
            )
            axis.plot(
                n_values,
                bias,
                color=color,
                marker="o",
                linewidth=2.0,
                label="Mean bias",
            )
            axis.set_xscale("log")
            axis.set_xticks(n_values, [str(n) for n in n_values])
            axis.set_xlabel("Fixed N pairs")
            axis.set_ylabel(f"{SCHEME_LABELS[scheme]}\nBias of {parameter}")
            axis.grid(alpha=0.35, linestyle=":")
            if row == 0:
                axis.set_title(parameter)
            if col == 0:
                axis.legend(fontsize=8)
    fig.suptitle("Posterior-mean bias (estimate - truth) by sample size")
    fig.tight_layout()
    fig.savefig(output_dir / "matched_bias_by_n.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_rank_plot(
    predictions: pd.DataFrame,
    schemes,
    n_pairs: int,
    output_dir: Path,
    n_bins: int = 10,
):
    """Plot SBC-style truth-percentile histograms for one fixed N."""
    fig, axes = plt.subplots(
        len(schemes), len(ACE_PARAM_NAMES), figsize=(15, 9), squeeze=False
    )
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for row, scheme in enumerate(schemes):
        subset = predictions.loc[
            (predictions["scheme"] == scheme)
            & (predictions["N_pairs"] == n_pairs)
        ]
        expected = len(subset) / n_bins
        for col, parameter in enumerate(ACE_PARAM_NAMES):
            axis = axes[row, col]
            axis.hist(
                subset[f"{parameter}_true_percentile"],
                bins=bins,
                color="steelblue",
                edgecolor="white",
            )
            axis.axhline(
                expected,
                color="red",
                linestyle="--",
                linewidth=1.25,
                label="Uniform expectation",
            )
            axis.set_xlim(0.0, 1.0)
            axis.set_xlabel("Posterior percentile of truth")
            axis.set_ylabel(f"{SCHEME_LABELS[scheme]}\nCount")
            axis.grid(axis="y", alpha=0.25)
            if row == 0:
                axis.set_title(parameter)
            if row == 0 and col == 0:
                axis.legend(fontsize=8)
    fig.suptitle(f"Truth-rank calibration, fixed N={n_pairs}")
    fig.tight_layout()
    fig.savefig(
        output_dir / f"matched_rank_histograms_N{n_pairs}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare matched train/test ACE schemes at each fixed N"
    )
    parser.add_argument("--models_dir", default="prior_comparison")
    parser.add_argument("--output_dir", default="prior_comparison")
    parser.add_argument(
        "--schemes", nargs="+", choices=SCHEMES, default=list(SCHEMES)
    )
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_posterior_samples", type=int, default=500)
    parser.add_argument("--n_pairs", type=int, nargs="+", default=None)
    parser.add_argument("--total_variance", type=float, default=1.0)
    parser.add_argument("--dirichlet_alpha", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    if args.n_samples <= 0 or args.n_posterior_samples <= 1:
        parser.error("--n_samples must be positive and --n_posterior_samples must exceed 1")
    torch.manual_seed(args.seed)
    n_options = tuple(args.n_pairs) if args.n_pairs is not None else DEFAULT_N_PAIRS
    if any(n < 2 for n in n_options):
        parser.error("--n_pairs values must be at least 2")

    models_dir = resolve(args.models_dir, MODELS_DIR)
    output_dir = resolve(args.output_dir, RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_data_dir = output_dir / "oos_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    test_sets = {}
    for n_pairs in n_options:
        for scheme in args.schemes:
            df = generate_dataset(
                n_samples=args.n_samples,
                scheme=scheme,
                n_pairs_options=[n_pairs],
                seed=args.seed,
                total_variance=args.total_variance,
                dirichlet_alpha=args.dirichlet_alpha,
            )
            df.to_csv(
                test_data_dir / f"oos_{scheme}_N{n_pairs}.csv", index=False
            )
            test_sets[(n_pairs, scheme)] = df

    loaded_models = {}
    for n_pairs in n_options:
        for scheme in args.schemes:
            loaded = load_posterior(model_dir_for(models_dir, scheme, n_pairs))
            config = loaded["config"]
            trained_scheme = config.get("simulation_scheme")
            trained_n = config.get("fixed_n_pairs")
            if trained_scheme != scheme or trained_n != n_pairs:
                raise ValueError(
                    f"Expected model ({scheme}, N={n_pairs}), but its config reports "
                    f"({trained_scheme}, N={trained_n})"
                )
            if config.get("n_is_model_feature") is not False:
                raise ValueError(
                    f"Model {scheme}, N={n_pairs} does not declare fixed-N training"
                )
            trained_total = config.get("total_variance")
            if trained_total is not None and not np.isclose(
                trained_total, args.total_variance
            ):
                raise ValueError(
                    f"Model {scheme!r}, N={n_pairs} was trained with V={trained_total}, "
                    f"but OOS generation requested V={args.total_variance}"
                )
            trained_alpha = config.get("dirichlet_alpha")
            if trained_alpha is not None and not np.allclose(
                trained_alpha, args.dirichlet_alpha
            ):
                raise ValueError(
                    f"Dirichlet model alpha={trained_alpha} does not match "
                    f"OOS alpha={args.dirichlet_alpha}"
                )
            loaded_models[(n_pairs, scheme)] = loaded

    metric_records, summary_records, prediction_frames = [], [], []
    for n_pairs in n_options:
        for scheme in args.schemes:
            print(f"Evaluating N={n_pairs} | matched={SCHEME_LABELS[scheme]}")
            evaluation = evaluate_model(
                loaded_models[(n_pairs, scheme)],
                test_sets[(n_pairs, scheme)],
                args.n_posterior_samples,
            )
            metric_records.extend(metric_rows(evaluation, scheme, n_pairs))
            summary_records.append(aggregate_row(evaluation, scheme, n_pairs))
            prediction_frames.append(
                prediction_frame(
                    evaluation,
                    scheme,
                    n_pairs,
                    test_sets[(n_pairs, scheme)],
                )
            )

    metrics = pd.DataFrame(metric_records)
    summary = pd.DataFrame(summary_records)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(output_dir / "matched_metrics_by_parameter.csv", index=False)
    summary.to_csv(output_dir / "matched_summary.csv", index=False)
    predictions.to_csv(output_dir / "matched_predictions.csv", index=False)
    with open(output_dir / "matched_config.json", "w") as f:
        json.dump(
            {
                "schemes": args.schemes,
                "evaluation_design": "matched_train_test_only",
                "n_samples_per_test_scheme": args.n_samples,
                "n_posterior_samples": args.n_posterior_samples,
                "posterior_kurtosis_definition": "Fisher excess kurtosis",
                "true_rank_definition": "number of posterior draws below truth",
                "true_percentile_definition": "true_rank / n_posterior_samples",
                "fixed_n_pairs": list(n_options),
                "one_model_per_n": True,
                "n_is_model_feature": False,
                "total_variance": args.total_variance,
                "dirichlet_alpha": args.dirichlet_alpha,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    for n_pairs in n_options:
        save_parameter_plot(metrics, args.schemes, n_pairs, output_dir)
        save_rank_plot(predictions, args.schemes, n_pairs, output_dir)
    save_matched_plot(metrics, args.schemes, n_options, output_dir)
    save_bias_plot(metrics, args.schemes, n_options, output_dir)

    print(f"\nSaved comparison tables and plots -> {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
