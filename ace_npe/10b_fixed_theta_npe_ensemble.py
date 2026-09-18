"""STEP 10b -- train and evaluate an ensemble of fixed-N NPEs.

The default experiment trains 100 independent Dirichlet NPEs.  Every NPE uses
100,000 simulations at N=1,000 and is evaluated on 100 datasets generated at
(A, C, E)=(0.4, 0.3, 0.3), with 2,000 posterior draws per dataset.

This file is designed for a Slurm array.  One array task runs one replicate::

    python 10b_fixed_theta_npe_ensemble.py run-one --replicate 1

After all array tasks finish, aggregate their summaries and make model-level
calibration plots plus dataset-level between-NPE uncertainty plots::

    python 10b_fixed_theta_npe_ensemble.py aggregate

Training covariance statistics are simulated directly from their exact
Wishart distribution.  This is distributionally identical to generating all
individual Gaussian twin observations and calling ``np.cov(..., ddof=1)``,
but avoids constructing 200 million observations per NPE.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_model import (  # noqa: E402
    ACE_PARAM_NAMES,
    COV_FEATURE_NAMES,
    RESULTS_DIR,
    VAR_REDUCTION,
    DirichletACEPosterior,
    DirichletALRPrior,
    ace_to_alr,
    resolve,
)


DEFAULT_OUTPUT_DIR = "fixed_theta_npe_ensemble"
DEFAULT_REPLICATES = 100
DEFAULT_TRAINING_SIMULATIONS = 100_000
DEFAULT_N_PAIRS = 1_000
DEFAULT_TEST_SIMULATIONS = 100
DEFAULT_POSTERIOR_DRAWS = 2_000
DEFAULT_THETA = (0.4, 0.3, 0.3)
DEFAULT_ALPHA = (1.0, 1.0, 1.0)
DEFAULT_SEED = 202_609_16
RMS_POSTERIOR_SE_COLUMN = "sqrt(mean(posterior_var))"
RMS_SE_RATIO_COLUMN = "sqrt(mean(posterior_var))/empirical_SE"


def integer_seed(base_seed: int, replicate: int, stream: int) -> int:
    """Return a reproducible uint32 seed for one experiment stream."""
    return int(
        np.random.SeedSequence([base_seed, replicate, stream]).generate_state(1)[0]
    )


def simulate_covariance_features(
    ace: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate exact Gaussian sample-covariance summaries in vectorized form.

    For ``n_pairs`` bivariate normal observations, the unbiased sample
    covariance satisfies ``(n_pairs - 1) S ~ Wishart_2(n_pairs - 1, Sigma)``.
    A vectorized 2-D Bartlett decomposition generates that distribution
    without storing the individual twin observations.
    """
    ace = np.asarray(ace, dtype=np.float64)
    if ace.ndim != 2 or ace.shape[1] != 3:
        raise ValueError("ace must have shape (n_simulations, 3)")
    if n_pairs < 3:
        raise ValueError("n_pairs must be at least 3")
    if np.any(ace <= 0):
        raise ValueError("A, C, and E must all be positive")

    variance = ace.sum(axis=1)
    mz_covariance = ace[:, 0] + ace[:, 1]
    dz_covariance = 0.5 * ace[:, 0] + ace[:, 1]
    degrees_freedom = n_pairs - 1

    def draw_one_group(off_diagonal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Cholesky factor of [[V, cov], [cov, V]] for every simulation.
        l00 = np.sqrt(variance)
        l10 = off_diagonal / l00
        remainder = variance - np.square(l10)
        if np.any(remainder <= 0):
            raise ValueError("Encountered a non-positive-definite ACE covariance")
        l11 = np.sqrt(remainder)

        # Bartlett factor A = [[sqrt(chi2_df), 0],
        #                       [N(0,1), sqrt(chi2_(df-1))]].
        a00 = np.sqrt(rng.chisquare(degrees_freedom, size=len(ace)))
        a10 = rng.standard_normal(len(ace))
        a11 = np.sqrt(rng.chisquare(degrees_freedom - 1, size=len(ace)))

        # B=L@A and W=B@B.T.  Only mean(diag(W)) and W[0,1] are needed.
        b00 = l00 * a00
        b10 = l10 * a00 + l11 * a10
        b11 = l11 * a11
        sample_var = (np.square(b00) + np.square(b10) + np.square(b11)) / (
            2.0 * degrees_freedom
        )
        sample_cov = (b00 * b10) / degrees_freedom
        return sample_var, sample_cov

    mz_var, mz_cov = draw_one_group(mz_covariance)
    dz_var, dz_cov = draw_one_group(dz_covariance)
    return np.column_stack((mz_var, mz_cov, dz_var, dz_cov)).astype(np.float32)


def make_training_data(args, replicate: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate one prior-predictive training corpus."""
    data_replicate = 0 if args.training_data_mode == "shared" else replicate
    seed = integer_seed(args.seed, data_replicate, 1)
    rng = np.random.default_rng(seed)
    alpha = np.asarray(args.dirichlet_alpha, dtype=np.float64)
    ace = rng.dirichlet(alpha, size=args.n_training_simulations)
    ace *= args.total_variance
    features = simulate_covariance_features(ace, args.n_pairs, rng)
    return features, ace.astype(np.float32), seed


def make_test_data(args, replicate: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate the fixed-theta evaluation datasets."""
    test_replicate = 0 if args.test_set_mode == "shared" else replicate
    seed = integer_seed(args.seed, test_replicate, 2)
    rng = np.random.default_rng(seed)
    theta = np.asarray(args.theta, dtype=np.float64)
    ace = np.repeat(theta[None, :], args.n_test_simulations, axis=0)
    return simulate_covariance_features(ace, args.n_pairs, rng), ace, seed


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return device


def train_posterior(
    args,
    replicate: int,
    features: np.ndarray,
    ace: np.ndarray,
    output_dir: Path,
):
    """Fit and save one fixed-N Dirichlet NPE."""
    split_seed = integer_seed(args.seed, replicate, 3)
    training_seed = integer_seed(args.seed, replicate, 4)
    rng = np.random.default_rng(split_seed)
    indices = rng.permutation(len(features))
    n_validation = max(1, int(round(args.validation_fraction * len(features))))
    n_training = len(features) - n_validation
    train_indices = indices[:n_training]
    validation_indices = indices[n_training:]

    scaler = StandardScaler().fit(features[train_indices])
    x_training = scaler.transform(features[train_indices]).astype(np.float32)
    x_validation = scaler.transform(features[validation_indices]).astype(np.float32)
    theta = ace_to_alr(ace)
    theta_all = np.vstack((theta[train_indices], theta[validation_indices]))
    x_all = np.vstack((x_training, x_validation))
    joblib.dump(scaler, output_dir / "feature_scaler.pkl")

    device = resolve_device(args.device)
    torch.manual_seed(training_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)

    prior = DirichletALRPrior(args.dirichlet_alpha, device=str(device))
    density_builder = posterior_nn(
        model=args.flow_type,
        embedding_net=nn.Identity(),
        hidden_features=args.flow_hidden,
        num_transforms=args.flow_transforms,
        z_score_theta="independent",
        z_score_x="independent",
    )
    inference = SNPE(
        prior=prior,
        density_estimator=density_builder,
        device=str(device),
    )
    inference.append_simulations(
        theta=torch.as_tensor(theta_all, dtype=torch.float32),
        x=torch.as_tensor(x_all, dtype=torch.float32),
    )
    estimator = inference.train(
        training_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_num_epochs=args.epochs,
        stop_after_epochs=args.stop_after_epochs,
        validation_fraction=n_validation / len(features),
        show_train_summary=True,
    )
    latent_posterior = inference.build_posterior(estimator)
    latent_posterior.to("cpu")
    posterior = DirichletACEPosterior(
        latent_posterior, total_variance=args.total_variance
    )

    torch.save(
        {
            "density_estimator_state_dict": estimator.state_dict(),
            "prior_type": "exact_dirichlet_ALR",
            "dirichlet_alpha": list(args.dirichlet_alpha),
            "total_variance": args.total_variance,
            "theta_parameterization": "ALR: log(A/E), log(C/E); E reconstructed",
        },
        output_dir / "density_estimator.pt",
    )
    with open(output_dir / "posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)
    return posterior, scaler, device, n_training, n_validation, training_seed


def evaluate_posterior(
    args,
    replicate: int,
    posterior,
    scaler: StandardScaler,
    raw_features: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one NPE and calculate the requested A/C/E coordinates."""
    scaled_features = scaler.transform(raw_features).astype(np.float32)
    posterior_seed = integer_seed(args.seed, replicate, 5)
    torch.manual_seed(posterior_seed)
    rows = []
    for test_index, (raw_row, scaled_row) in enumerate(
        zip(raw_features, scaled_features), start=1
    ):
        x = torch.as_tensor(scaled_row, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            samples = posterior.sample(
                (args.n_posterior_draws,), x=x, show_progress_bars=False
            )
        draws = samples.cpu().numpy()
        row = {
            "replicate": replicate,
            "test_simulation": test_index,
            **dict(zip(COV_FEATURE_NAMES, raw_row.astype(float))),
        }
        for parameter_index, parameter in enumerate(ACE_PARAM_NAMES):
            row[f"true_{parameter}"] = float(args.theta[parameter_index])
            row[f"{parameter}_posterior_mean"] = float(
                draws[:, parameter_index].mean()
            )
            row[f"{parameter}_posterior_sd"] = float(
                draws[:, parameter_index].std(ddof=1)
            )
        rows.append(row)

    raw = pd.DataFrame(rows)
    summary_rows = []
    for parameter in ACE_PARAM_NAMES:
        posterior_sds = raw[f"{parameter}_posterior_sd"]
        summary_rows.append(
            {
                "replicate": replicate,
                "parameter": parameter,
                "n_test_simulations": len(raw),
                "SE(mean(theta))": raw[f"{parameter}_posterior_mean"].std(ddof=1),
                "mean(posterior_SE)": posterior_sds.mean(),
                RMS_POSTERIOR_SE_COLUMN: np.sqrt(np.mean(np.square(posterior_sds))),
                "mean(posterior_mean)": raw[f"{parameter}_posterior_mean"].mean(),
                "bias(posterior_mean)": (
                    raw[f"{parameter}_posterior_mean"].mean()
                    - float(args.theta[ACE_PARAM_NAMES.index(parameter)])
                ),
            }
        )
    return raw, pd.DataFrame(summary_rows)


def run_one(args) -> None:
    replicate = args.replicate
    output_root = resolve(args.output_dir, RESULTS_DIR)
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / f"replicate_{replicate:03d}"
    complete_path = final_dir / "COMPLETE"
    if complete_path.exists() and not args.force:
        print(f"Replicate {replicate} is already complete: {final_dir}")
        return
    if final_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Incomplete output directory already exists: {final_dir}. "
                "Inspect it, then rerun with --force to replace it."
            )
        shutil.rmtree(final_dir)

    work_dir = output_root / f".replicate_{replicate:03d}.tmp.{os.getpid()}"
    work_dir.mkdir(parents=False, exist_ok=False)
    start = time.perf_counter()
    try:
        print(f"Replicate {replicate}: generating training simulations ...")
        training_features, training_ace, training_data_seed = make_training_data(
            args, replicate
        )
        print(
            f"Replicate {replicate}: training on "
            f"{args.n_training_simulations:,} simulations ..."
        )
        posterior, scaler, device, n_training, n_validation, training_seed = (
            train_posterior(
                args,
                replicate,
                training_features,
                training_ace,
                work_dir,
            )
        )
        # Release the largest arrays before posterior evaluation.
        del training_features, training_ace

        print(f"Replicate {replicate}: generating and evaluating fixed-theta data ...")
        test_features, _, test_data_seed = make_test_data(args, replicate)
        raw, summary = evaluate_posterior(
            args, replicate, posterior, scaler, test_features
        )
        raw.to_csv(work_dir / "fixed_theta_posterior_results.csv", index=False)
        summary.to_csv(work_dir / "fixed_theta_summary.csv", index=False)

        elapsed_seconds = time.perf_counter() - start
        config = {
            "experiment": "fixed_theta_npe_ensemble",
            "model_type": "NPE_fixed_theta_ensemble",
            "simulation_scheme": "dirichlet",
            "replicate": replicate,
            "n_replicates_planned": args.n_replicates,
            "n_training_simulations_total": args.n_training_simulations,
            "training_rows": n_training,
            "validation_rows": n_validation,
            "training_data_mode": args.training_data_mode,
            "test_set_mode": args.test_set_mode,
            "fixed_n_pairs": args.n_pairs,
            "n_is_model_feature": False,
            "theta": dict(zip(ACE_PARAM_NAMES, args.theta)),
            "n_test_simulations": args.n_test_simulations,
            "n_posterior_draws": args.n_posterior_draws,
            "dirichlet_alpha": list(args.dirichlet_alpha),
            "total_variance": args.total_variance,
            "simulation_method": "exact vectorized Wishart_2 Bartlett decomposition",
            "feature_cols": list(COV_FEATURE_NAMES),
            "param_names": list(ACE_PARAM_NAMES),
            "var_feature": VAR_REDUCTION,
            "flow_type": args.flow_type,
            "flow_hidden": args.flow_hidden,
            "flow_transforms": args.flow_transforms,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_epochs": args.epochs,
            "stop_after_epochs": args.stop_after_epochs,
            "device": str(device),
            "base_seed": args.seed,
            "training_data_seed": training_data_seed,
            "training_seed": training_seed,
            "test_data_seed": test_data_seed,
            "elapsed_seconds": elapsed_seconds,
            "definition": {
                "SE(mean(theta))": (
                    "sample SD (ddof=1) of posterior means over fixed-theta "
                    "test datasets"
                ),
                "mean(posterior_SE)": (
                    "mean posterior sample SD (ddof=1) over fixed-theta "
                    "test datasets"
                ),
                RMS_POSTERIOR_SE_COLUMN: (
                    "square root of the mean posterior sample variance over "
                    "fixed-theta test datasets; primary calibration metric"
                ),
            },
        }
        with open(work_dir / "config.json", "w") as handle:
            json.dump(config, handle, indent=2)
        (work_dir / "COMPLETE").write_text("complete\n")
        work_dir.rename(final_dir)
        print(
            f"Replicate {replicate} complete in {elapsed_seconds / 60:.1f} min: "
            f"{final_dir}"
        )
    except Exception:
        print(f"Partial files retained for diagnosis in {work_dir}", file=sys.stderr)
        raise


def save_ensemble_plot(
    summary: pd.DataFrame,
    theta: np.ndarray,
    n_pairs: int,
    path: Path,
) -> None:
    test_counts = summary["n_test_simulations"].drop_duplicates().tolist()
    if len(test_counts) != 1:
        raise ValueError(f"Replicates disagree on test-set size: {test_counts}")
    n_test_simulations = int(test_counts[0])
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1), squeeze=False)
    colors = {"A": "tab:blue", "C": "tab:orange", "E": "tab:green"}
    for column, parameter in enumerate(ACE_PARAM_NAMES):
        axis = axes[0, column]
        selected = summary.loc[summary["parameter"] == parameter]
        x = selected["SE(mean(theta))"].to_numpy()
        y = selected[RMS_POSTERIOR_SE_COLUMN].to_numpy()
        combined = np.concatenate((x, y))
        lower = float(combined.min())
        upper = float(combined.max())
        span = upper - lower
        if span == 0:
            span = max(abs(upper), 1.0) * 0.05
        padding = 0.08 * span
        lower = max(0.0, lower - padding)
        upper += padding
        axis.scatter(
            x,
            y,
            s=52,
            alpha=0.72,
            color=colors[parameter],
            edgecolor="white",
            linewidth=0.45,
        )
        axis.plot(
            [lower, upper], [lower, upper], "--", color="0.35", linewidth=1.2
        )
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{parameter} (true {theta[column]:g})")
        axis.set_xlabel(
            f"sqrt(B_m): SD of posterior means across {n_test_simulations} datasets"
        )
        axis.set_ylabel("sqrt(W_m): RMS posterior SD")
        axis.grid(alpha=0.22)

    fig.suptitle(
        f"{summary['replicate'].nunique()} independently trained NPEs "
        f"with N pairs={n_pairs}\n"
        "Each point is one NPE; dashed line: "
        "sqrt(W_m) = sqrt(B_m)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_dataset_model_uncertainty(
    raw_results: pd.DataFrame,
    n_replicates: int,
) -> pd.DataFrame:
    """Calculate between-NPE and within-posterior variance for each dataset.

    For shared test dataset ``j`` and parameter ``theta``:

      B_j = Var_m(E[theta | x_j, NPE_m])
      W_j = Mean_m(Var[theta | x_j, NPE_m])

    Thus B_j isolates disagreement among independently trained NPEs while
    W_j is their average reported posterior variance for the same dataset.
    """
    if n_replicates < 2:
        raise ValueError("At least two NPE replicates are required for B_j")

    rows = []
    for parameter in ACE_PARAM_NAMES:
        mean_column = f"{parameter}_posterior_mean"
        sd_column = f"{parameter}_posterior_sd"
        for test_simulation, selected in raw_results.groupby(
            "test_simulation", sort=True
        ):
            if len(selected) != n_replicates:
                raise ValueError(
                    f"Test dataset {test_simulation} has {len(selected)} NPE "
                    f"results; expected {n_replicates}"
                )
            posterior_means = selected[mean_column].to_numpy(dtype=float)
            posterior_variances = np.square(
                selected[sd_column].to_numpy(dtype=float)
            )
            between_npe_variance = float(posterior_means.var(ddof=1))
            within_posterior_variance = float(posterior_variances.mean())
            if between_npe_variance <= 0 or within_posterior_variance <= 0:
                raise ValueError(
                    f"Non-positive B_j or W_j for {parameter}, dataset "
                    f"{test_simulation}"
                )
            rows.append(
                {
                    "test_simulation": int(test_simulation),
                    "parameter": parameter,
                    "n_models": n_replicates,
                    "B_j_between_NPE_variance": between_npe_variance,
                    "W_j_mean_posterior_variance": within_posterior_variance,
                    "sqrt(B_j)": np.sqrt(between_npe_variance),
                    "sqrt(W_j)": np.sqrt(within_posterior_variance),
                    "B_j/W_j": (
                        between_npe_variance / within_posterior_variance
                    ),
                    "mean_NPE_posterior_mean": float(posterior_means.mean()),
                }
            )
    return pd.DataFrame(rows)


def save_dataset_model_uncertainty_scatter(
    dataset_summary: pd.DataFrame,
    theta: np.ndarray,
    n_pairs: int,
    path: Path,
) -> None:
    """Compare between-NPE and within-posterior SD for each shared dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1), squeeze=False)
    colors = {"A": "tab:blue", "C": "tab:orange", "E": "tab:green"}
    for column, parameter in enumerate(ACE_PARAM_NAMES):
        axis = axes[0, column]
        selected = dataset_summary.loc[dataset_summary["parameter"] == parameter]
        x = selected["sqrt(B_j)"].to_numpy()
        y = selected["sqrt(W_j)"].to_numpy()
        if np.any(x <= 0) or np.any(y <= 0):
            raise ValueError("Dataset-level uncertainty values must be positive")

        def padded_limits(values: np.ndarray) -> tuple[float, float]:
            lower = float(values.min())
            upper = float(values.max())
            span = upper - lower
            if span == 0:
                span = max(abs(upper), 1.0) * 0.05
            return max(0.0, lower - 0.10 * span), upper + 0.10 * span

        x_lower, x_upper = padded_limits(x)
        y_lower, y_upper = padded_limits(y)

        axis.scatter(
            x,
            y,
            s=48,
            alpha=0.68,
            color=colors[parameter],
            edgecolor="white",
            linewidth=0.4,
        )
        axis.set_xlim(x_lower, x_upper)
        axis.set_ylim(y_lower, y_upper)
        equality_lower = max(x_lower, y_lower)
        equality_upper = min(x_upper, y_upper)
        if equality_lower < equality_upper:
            axis.plot(
                [equality_lower, equality_upper],
                [equality_lower, equality_upper],
                "--",
                color="0.35",
                linewidth=1.2,
            )
        axis.set_title(f"{parameter} (true {theta[column]:g})")
        axis.set_xlabel("SD of posterior means across NPEs")
        axis.set_ylabel("RMS posterior SD across NPEs")
        axis.grid(alpha=0.22, which="both")

    n_models = int(dataset_summary["n_models"].drop_duplicates().item())
    n_datasets = int(dataset_summary["test_simulation"].nunique())
    fig.suptitle(
        f"Between-NPE versus within-posterior uncertainty; N pairs={n_pairs}\n"
        f"Each point is one shared dataset ({n_models} NPEs; "
        f"{n_datasets} datasets); axes are tightly scaled to show variation"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_dataset_model_uncertainty_ratio_boxplot(
    dataset_summary: pd.DataFrame,
    n_pairs: int,
    path: Path,
) -> None:
    """Plot B_j/W_j across shared datasets with every dataset visible."""
    values_by_parameter = [
        dataset_summary.loc[
            dataset_summary["parameter"] == parameter, "B_j/W_j"
        ].to_numpy()
        for parameter in ACE_PARAM_NAMES
    ]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, axis = plt.subplots(figsize=(8.2, 5.8))
    boxes = axis.boxplot(
        values_by_parameter,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "0.15", "linewidth": 1.8},
        whiskerprops={"color": "0.35", "linewidth": 1.2},
        capprops={"color": "0.35", "linewidth": 1.2},
    )
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.28)
        box.set_edgecolor(color)
        box.set_linewidth(1.4)

    rng = np.random.default_rng(20260918)
    for position, (values, color) in enumerate(
        zip(values_by_parameter, colors), start=1
    ):
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        axis.scatter(
            position + jitter,
            values,
            s=28,
            alpha=0.58,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )

    all_values = np.concatenate(values_by_parameter)
    if np.any(all_values <= 0):
        raise ValueError("B_j/W_j ratios must be positive")
    log_lower = float(np.log10(all_values.min()))
    log_upper = float(np.log10(all_values.max()))
    log_span = log_upper - log_lower
    if log_span == 0:
        log_span = 0.1
    y_lower = 10 ** (log_lower - 0.10 * log_span)
    y_upper = 10 ** (log_upper + 0.10 * log_span)
    axis.set_yscale("log")
    axis.set_ylim(y_lower, y_upper)
    if y_lower <= 1.0 <= y_upper:
        axis.axhline(
            1.0,
            linestyle="--",
            color="0.35",
            linewidth=1.3,
            label="Between-NPE variance = posterior variance",
            zorder=1,
        )
        axis.legend(frameon=False, loc="best")
    elif y_upper < 1.0:
        axis.text(
            0.99,
            0.98,
            "B_j / W_j = 1 lies above the displayed range",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="0.35",
        )
    else:
        axis.text(
            0.99,
            0.02,
            "B_j / W_j = 1 lies below the displayed range",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="0.35",
        )
    tick_labels = [
        f"{parameter}\nmedian = {np.median(values):.3g}"
        for parameter, values in zip(ACE_PARAM_NAMES, values_by_parameter)
    ]
    axis.set_xticks(range(1, len(ACE_PARAM_NAMES) + 1), tick_labels)
    axis.set_xlabel("ACE parameter")
    axis.set_ylabel("B_j / W_j (log scale)")
    axis.set_title(
        f"Between-NPE model variance relative to posterior variance; "
        f"N pairs={n_pairs}\n"
        "Each point is one shared test dataset"
    )
    axis.grid(axis="y", alpha=0.22, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_se_ratio_boxplot(
    summary: pd.DataFrame,
    n_pairs: int,
    path: Path,
) -> None:
    """Show model-to-model calibration ratios with boxes and all NPE points."""
    values_by_parameter = [
        summary.loc[
            summary["parameter"] == parameter,
            RMS_SE_RATIO_COLUMN,
        ].to_numpy()
        for parameter in ACE_PARAM_NAMES
    ]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axis = plt.subplots(figsize=(8.2, 5.8))
    boxes = axis.boxplot(
        values_by_parameter,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "0.15", "linewidth": 1.8},
        whiskerprops={"color": "0.35", "linewidth": 1.2},
        capprops={"color": "0.35", "linewidth": 1.2},
    )
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.28)
        box.set_edgecolor(color)
        box.set_linewidth(1.4)

    # A deterministic horizontal jitter exposes the full distribution without
    # allowing overlapping points to hide how concentrated the NPEs are.
    rng = np.random.default_rng(20260917)
    for position, (parameter, values, color) in enumerate(
        zip(ACE_PARAM_NAMES, values_by_parameter, colors), start=1
    ):
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        axis.scatter(
            position + jitter,
            values,
            s=28,
            alpha=0.55,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    all_values = np.concatenate(values_by_parameter)
    lower = min(float(all_values.min()), 1.0)
    upper = max(float(all_values.max()), 1.0)
    padding = max(0.015, 0.12 * (upper - lower))
    axis.set_ylim(max(0.0, lower - padding), upper + padding)
    axis.axhline(
        1.0,
        linestyle="--",
        color="0.35",
        linewidth=1.3,
        label="Perfect calibration (ratio = 1)",
        zorder=1,
    )
    tick_labels = [
        f"{parameter}\nmedian = {np.median(values):.3f}"
        for parameter, values in zip(ACE_PARAM_NAMES, values_by_parameter)
    ]
    axis.set_xticks(range(1, len(ACE_PARAM_NAMES) + 1), tick_labels)
    axis.set_xlabel("ACE parameter")
    axis.set_ylabel("sqrt(mean posterior variance) / empirical SE")
    axis.set_title(
        f"{summary['replicate'].nunique()} Independently trained NPE "
        f"with N pairs={n_pairs}\n"
        "RMS posterior-to-empirical SE ratio across models"
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def aggregate(args) -> None:
    output_root = resolve(args.output_dir, RESULTS_DIR)
    frames = []
    raw_frames = []
    reference_test_data = None
    missing = []
    for replicate in range(1, args.n_replicates + 1):
        replicate_dir = output_root / f"replicate_{replicate:03d}"
        summary_path = replicate_dir / "fixed_theta_summary.csv"
        if not (replicate_dir / "COMPLETE").exists() or not summary_path.exists():
            missing.append(replicate)
            continue
        frame = pd.read_csv(summary_path)
        if set(frame["parameter"]) != set(ACE_PARAM_NAMES) or len(frame) != 3:
            raise ValueError(f"Malformed replicate summary: {summary_path}")
        raw_path = replicate_dir / "fixed_theta_posterior_results.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing per-dataset results: {raw_path}")
        raw = pd.read_csv(raw_path)
        required_raw_columns = {
            "replicate",
            "test_simulation",
            *COV_FEATURE_NAMES,
            *(f"true_{parameter}" for parameter in ACE_PARAM_NAMES),
            *(f"{parameter}_posterior_mean" for parameter in ACE_PARAM_NAMES),
            *(f"{parameter}_posterior_sd" for parameter in ACE_PARAM_NAMES),
        }
        missing_raw_columns = required_raw_columns.difference(raw.columns)
        if missing_raw_columns:
            raise ValueError(
                f"Missing columns in {raw_path}: {sorted(missing_raw_columns)}"
            )
        if raw["test_simulation"].duplicated().any():
            raise ValueError(f"Duplicate test_simulation values in {raw_path}")
        raw = raw.sort_values("test_simulation").reset_index(drop=True)
        raw["replicate"] = replicate

        shared_columns = [
            "test_simulation",
            *COV_FEATURE_NAMES,
            *(f"true_{parameter}" for parameter in ACE_PARAM_NAMES),
        ]
        current_test_data = raw[shared_columns]
        if reference_test_data is None:
            reference_test_data = current_test_data.copy()
        else:
            if not np.array_equal(
                current_test_data["test_simulation"].to_numpy(),
                reference_test_data["test_simulation"].to_numpy(),
            ):
                raise ValueError(
                    f"Test simulation IDs differ in replicate {replicate}"
                )
            numeric_columns = shared_columns[1:]
            if not np.allclose(
                current_test_data[numeric_columns].to_numpy(dtype=float),
                reference_test_data[numeric_columns].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-10,
            ):
                raise ValueError(
                    "Dataset-level B_j/W_j requires the shared STEP 10b test "
                    f"set, but replicate {replicate} contains different data"
                )

        # Runs completed before the RMS-SE metric was introduced already have
        # every per-dataset posterior SD in their raw results. Reconstruct the
        # metric here so aggregation never requires retraining those NPEs.
        if RMS_POSTERIOR_SE_COLUMN not in frame:
            frame[RMS_POSTERIOR_SE_COLUMN] = np.nan
            for parameter in ACE_PARAM_NAMES:
                sd_column = f"{parameter}_posterior_sd"
                if sd_column not in raw:
                    raise ValueError(f"Missing {sd_column} in {raw_path}")
                posterior_sds = raw[sd_column].to_numpy(dtype=float)
                if len(posterior_sds) < 2 or not np.isfinite(posterior_sds).all():
                    raise ValueError(f"Invalid posterior SD values in {raw_path}")
                frame.loc[
                    frame["parameter"] == parameter,
                    RMS_POSTERIOR_SE_COLUMN,
                ] = np.sqrt(np.mean(np.square(posterior_sds)))
        frames.append(frame)
        raw_frames.append(raw)
    if missing:
        raise RuntimeError(
            f"Cannot aggregate: {len(missing)} replicate(s) are incomplete: {missing}"
        )

    summary = pd.concat(frames, ignore_index=True)
    raw_results = pd.concat(raw_frames, ignore_index=True)
    if (summary["SE(mean(theta))"] <= 0).any():
        raise ValueError("Empirical SE must be positive to calculate SE ratios")
    summary[RMS_SE_RATIO_COLUMN] = (
        summary[RMS_POSTERIOR_SE_COLUMN] / summary["SE(mean(theta))"]
    )
    long_path = output_root / "ensemble_summary_long.csv"
    summary.to_csv(long_path, index=False)
    wide = summary.pivot(
        index="replicate",
        columns="parameter",
        values=[
            "SE(mean(theta))",
            "mean(posterior_SE)",
            RMS_POSTERIOR_SE_COLUMN,
            RMS_SE_RATIO_COLUMN,
        ],
    )
    wide = wide.swaplevel(0, 1, axis=1).reindex(columns=ACE_PARAM_NAMES, level=0)
    wide.columns = [f"{parameter}: {metric}" for parameter, metric in wide.columns]
    wide.reset_index().to_csv(output_root / "ensemble_summary_table.csv", index=False)

    ratio_summary = (
        summary.groupby("parameter", sort=False)[RMS_SE_RATIO_COLUMN]
        .agg(
            n_models="count",
            mean="mean",
            sd="std",
            minimum="min",
            q1=lambda values: values.quantile(0.25),
            median="median",
            q3=lambda values: values.quantile(0.75),
            maximum="max",
        )
        .reindex(ACE_PARAM_NAMES)
        .reset_index()
    )
    ratio_summary_path = output_root / "ensemble_se_ratio_summary.csv"
    ratio_summary.to_csv(ratio_summary_path, index=False)

    dataset_summary = summarize_dataset_model_uncertainty(
        raw_results, args.n_replicates
    )
    dataset_summary_path = output_root / "ensemble_dataset_uncertainty.csv"
    dataset_summary.to_csv(dataset_summary_path, index=False)

    theta = np.asarray(args.theta, dtype=float)
    plot_path = output_root / "fixed_theta_npe_scatter.png"
    save_ensemble_plot(summary, theta, args.n_pairs, plot_path)
    ratio_plot_path = output_root / "fixed_theta_npe_se_ratio_boxplot.png"
    save_se_ratio_boxplot(summary, args.n_pairs, ratio_plot_path)
    dataset_scatter_path = (
        output_root / "fixed_theta_npe_dataset_model_uncertainty_scatter.png"
    )
    save_dataset_model_uncertainty_scatter(
        dataset_summary, theta, args.n_pairs, dataset_scatter_path
    )
    dataset_ratio_path = (
        output_root / "fixed_theta_npe_dataset_B_over_W_boxplot.png"
    )
    save_dataset_model_uncertainty_ratio_boxplot(
        dataset_summary, args.n_pairs, dataset_ratio_path
    )
    print(f"Aggregated {args.n_replicates} NPEs")
    print(f"Summary: {long_path}")
    print(f"Ratio summary: {ratio_summary_path}")
    print(f"Dataset uncertainty: {dataset_summary_path}")
    print(f"Scatter plot:  {plot_path}")
    print(f"Ratio plot:    {ratio_plot_path}")
    print(f"Dataset scatter: {dataset_scatter_path}")
    print(f"Dataset B/W:     {dataset_ratio_path}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n_replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--n_pairs", type=int, default=DEFAULT_N_PAIRS)
    parser.add_argument("--theta", type=float, nargs=3, default=list(DEFAULT_THETA))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)


def validate_common(parser: argparse.ArgumentParser, args) -> None:
    if args.n_replicates < 1:
        parser.error("--n_replicates must be positive")
    if args.n_pairs < 3:
        parser.error("--n_pairs must be at least 3")
    theta = np.asarray(args.theta, dtype=float)
    if np.any(theta <= 0) or not np.isclose(theta.sum(), 1.0):
        parser.error("--theta must contain three positive values summing to 1")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-one", help="Train/evaluate one NPE")
    add_common_arguments(run_parser)
    run_parser.add_argument("--replicate", type=int, required=True)
    run_parser.add_argument(
        "--n_training_simulations", type=int, default=DEFAULT_TRAINING_SIMULATIONS
    )
    run_parser.add_argument(
        "--n_test_simulations", type=int, default=DEFAULT_TEST_SIMULATIONS
    )
    run_parser.add_argument(
        "--n_posterior_draws", type=int, default=DEFAULT_POSTERIOR_DRAWS
    )
    run_parser.add_argument(
        "--training_data_mode", choices=("independent", "shared"), default="independent"
    )
    run_parser.add_argument(
        "--test_set_mode", choices=("shared", "independent"), default="shared"
    )
    run_parser.add_argument(
        "--dirichlet_alpha", type=float, nargs=3, default=list(DEFAULT_ALPHA)
    )
    run_parser.add_argument("--total_variance", type=float, default=1.0)
    run_parser.add_argument("--validation_fraction", type=float, default=0.15)
    run_parser.add_argument("--epochs", type=int, default=500)
    run_parser.add_argument("--stop_after_epochs", type=int, default=50)
    run_parser.add_argument("--batch_size", type=int, default=1024)
    run_parser.add_argument("--learning_rate", type=float, default=5e-4)
    run_parser.add_argument(
        "--flow_type", choices=("nsf", "maf", "maf_rqs", "mdn"), default="nsf"
    )
    run_parser.add_argument("--flow_hidden", type=int, default=64)
    run_parser.add_argument("--flow_transforms", type=int, default=5)
    run_parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    run_parser.add_argument("--force", action="store_true")

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Combine completed replicates and plot"
    )
    add_common_arguments(aggregate_parser)

    args = parser.parse_args()
    selected_parser = run_parser if args.command == "run-one" else aggregate_parser
    validate_common(selected_parser, args)
    if args.command == "run-one":
        if not 1 <= args.replicate <= args.n_replicates:
            run_parser.error("--replicate must be between 1 and --n_replicates")
        if args.n_training_simulations < 20:
            run_parser.error("--n_training_simulations must be at least 20")
        if args.n_test_simulations < 2 or args.n_posterior_draws < 2:
            run_parser.error("test simulations and posterior draws must be at least 2")
        if not 0 < args.validation_fraction < 1:
            run_parser.error("--validation_fraction must be between 0 and 1")
        if any(alpha <= 0 for alpha in args.dirichlet_alpha):
            run_parser.error("--dirichlet_alpha values must be positive")
        if args.total_variance <= 0:
            run_parser.error("--total_variance must be positive")
        if not np.isclose(sum(args.theta), args.total_variance):
            run_parser.error("sum(--theta) must equal --total_variance")
        if min(args.epochs, args.stop_after_epochs, args.batch_size) <= 0:
            run_parser.error("epochs, patience, and batch size must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "run-one":
        cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        torch.set_num_threads(max(1, cpu_count))
        run_one(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
