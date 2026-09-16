"""Evaluate the existing fixed-N=1000 Dirichlet NPE at one ACE condition.

The default experiment generates 500 independent MZ/DZ twin datasets from
theta=(A, C, E)=(0.4, 0.3, 0.3), evaluates the already-trained NPE for every
dataset, and compares:

* the empirical SE of the posterior-mean estimator (the sample SD of posterior
  means across repeated datasets); and
* the mean posterior SE (the average posterior SD reported by the NPE).

Metrics are reported for all 500 simulations and for five non-overlapping
blocks of 100. No NPE is trained or modified by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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


DEFAULT_MODEL_DIR = "prior_comparison/dirichlet/N1000"
DEFAULT_OUTPUT_DIR = "fixed_theta_n1000"


def validate_model(loaded: dict, n_pairs: int) -> None:
    """Ensure the loaded run is the requested fixed-N Dirichlet model."""
    config = loaded["config"]
    if config.get("simulation_scheme") != "dirichlet":
        raise ValueError("The selected model is not a Dirichlet NPE")
    if config.get("fixed_n_pairs") != n_pairs:
        raise ValueError(
            f"The selected model was trained at fixed N="
            f"{config.get('fixed_n_pairs')}, not N={n_pairs}"
        )
    if config.get("n_is_model_feature") is not False:
        raise ValueError("Expected a fixed-N model with no N feature")
    if list(loaded["feature_cols"]) != list(COV_FEATURE_NAMES):
        raise ValueError(
            f"Model features are {loaded['feature_cols']}; expected "
            f"{list(COV_FEATURE_NAMES)}"
        )
    if list(loaded["param_names"]) != list(ACE_PARAM_NAMES):
        raise ValueError(
            f"Model parameters are {loaded['param_names']}; expected "
            f"{list(ACE_PARAM_NAMES)}"
        )


def simulate_features(
    theta: np.ndarray,
    n_pairs: int,
    n_simulations: int,
    seed: int,
) -> np.ndarray:
    """Simulate independent covariance-summary feature vectors."""
    mz_population, dz_population = theoretical_covariances(*theta)
    features = np.empty((n_simulations, len(COV_FEATURE_NAMES)), dtype=np.float32)

    for simulation_index in range(n_simulations):
        # Separate deterministic streams for MZ and DZ data make every row
        # reproducible without relying on NumPy's global RNG state.
        mz_rng = np.random.default_rng(
            np.random.SeedSequence([seed, simulation_index + 1, 1])
        )
        dz_rng = np.random.default_rng(
            np.random.SeedSequence([seed, simulation_index + 1, 2])
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
    n_posterior_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Return posterior means and SDs for every simulated dataset."""
    scaled_features = loaded["scaler"].transform(raw_features).astype(np.float32)
    posterior = loaded["posterior"]
    torch.manual_seed(seed)

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

        row = {
            "simulation": simulation_index,
            **{
                feature: float(value)
                for feature, value in zip(COV_FEATURE_NAMES, raw_features[simulation_index - 1])
            },
        }
        for parameter_index, parameter in enumerate(ACE_PARAM_NAMES):
            row[f"true_{parameter}"] = float(theta[parameter_index])
            row[f"{parameter}_posterior_mean"] = float(
                posterior_mean[parameter_index]
            )
            row[f"{parameter}_posterior_sd"] = float(posterior_sd[parameter_index])
        rows.append(row)

        if simulation_index % progress_every == 0 or simulation_index == total:
            print(f"  Evaluated {simulation_index}/{total} simulated datasets")

    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame, group_size: int) -> pd.DataFrame:
    """Summarize all simulations and consecutive non-overlapping blocks."""
    groups: list[tuple[str, pd.DataFrame]] = [(f"All {len(results)}", results)]
    for start in range(0, len(results), group_size):
        stop = min(start + group_size, len(results))
        groups.append((f"{start + 1}-{stop}", results.iloc[start:stop]))

    rows = []
    for group_label, group in groups:
        for parameter in ACE_PARAM_NAMES:
            rows.append(
                {
                    "group": group_label,
                    "n_simulations": len(group),
                    "parameter": parameter,
                    # This is the repository's mc_se: the empirical sampling
                    # SD of the posterior-mean estimator, not SD/sqrt(R).
                    "SE(mean(theta))": group[f"{parameter}_posterior_mean"].std(
                        ddof=1
                    ),
                    "mean(posterior_SE)": group[f"{parameter}_posterior_sd"].mean(),
                }
            )
    return pd.DataFrame(rows)


def make_wide_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Format the requested A/C/E pairs as one compact printed table."""
    group_order = {
        group: position
        for position, group in enumerate(summary["group"].drop_duplicates())
    }
    wide = summary.pivot(
        index=["group", "n_simulations"],
        columns="parameter",
        values=["SE(mean(theta))", "mean(posterior_SE)"],
    )
    wide = wide.swaplevel(0, 1, axis=1).reindex(columns=ACE_PARAM_NAMES, level=0)
    wide.columns = [f"{parameter}: {metric}" for parameter, metric in wide.columns]
    wide = wide.reset_index()
    wide["_group_order"] = wide["group"].map(group_order)
    return wide.sort_values("_group_order").drop(columns="_group_order").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed-N Dirichlet NPE at one fixed ACE condition"
    )
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--n_pairs", type=int, default=1000)
    parser.add_argument("--theta", type=float, nargs=3, default=[0.4, 0.3, 0.3])
    parser.add_argument("--n_simulations", type=int, default=500)
    parser.add_argument("--group_size", type=int, default=100)
    parser.add_argument("--n_posterior_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.n_pairs < 2:
        parser.error("--n_pairs must be at least 2")
    if args.n_simulations < 2:
        parser.error("--n_simulations must be at least 2")
    if args.group_size < 2:
        parser.error("--group_size must be at least 2")
    if args.n_posterior_samples < 2:
        parser.error("--n_posterior_samples must be at least 2")

    theta = np.asarray(args.theta, dtype=float)
    if theta.shape != (3,) or np.any(theta <= 0):
        parser.error("--theta must contain three positive values")
    if not np.isclose(theta.sum(), 1.0):
        parser.error("The fixed-N Dirichlet NPE requires A+C+E=1")

    model_dir = resolve(args.model_dir, MODELS_DIR)
    output_dir = resolve(args.output_dir, RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading the existing NPE (no retraining) ...")
    loaded = load_posterior(model_dir)
    validate_model(loaded, args.n_pairs)
    print(f"  Model: {model_dir}")
    print(
        f"Simulating {args.n_simulations} datasets at N={args.n_pairs}, "
        f"theta=({theta[0]}, {theta[1]}, {theta[2]}) ..."
    )
    raw_features = simulate_features(
        theta=theta,
        n_pairs=args.n_pairs,
        n_simulations=args.n_simulations,
        seed=args.seed,
    )
    results = evaluate_posterior(
        loaded=loaded,
        raw_features=raw_features,
        theta=theta,
        n_posterior_samples=args.n_posterior_samples,
        seed=args.seed,
    )
    summary = summarize_results(results, args.group_size)
    wide_summary = make_wide_table(summary)

    raw_path = output_dir / "fixed_theta_n1000_posterior_results.csv"
    summary_path = output_dir / "fixed_theta_n1000_summary_long.csv"
    table_path = output_dir / "fixed_theta_n1000_summary_table.csv"
    config_path = output_dir / "config.json"
    results.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    wide_summary.to_csv(table_path, index=False)
    with open(config_path, "w") as handle:
        json.dump(
            {
                "model_dir": str(model_dir),
                "n_pairs": args.n_pairs,
                "theta": dict(zip(ACE_PARAM_NAMES, theta.tolist())),
                "n_simulations": args.n_simulations,
                "group_size": args.group_size,
                "n_posterior_samples": args.n_posterior_samples,
                "seed": args.seed,
                "definition": {
                    "SE(mean(theta))": (
                        "sample SD (ddof=1) of posterior means across simulated datasets"
                    ),
                    "mean(posterior_SE)": (
                        "mean posterior sample SD (ddof=1) across simulated datasets"
                    ),
                },
            },
            handle,
            indent=2,
        )

    print("\nSummary:")
    print(wide_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nRaw results:  {raw_path}")
    print(f"Summary table: {table_path}")


if __name__ == "__main__":
    main()
