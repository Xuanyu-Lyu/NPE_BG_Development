"""STEP 11: compare AE and ACE NPE posterior predictive fit at fixed N.

Run --model ae and --model ace as independent RC array tasks, then --plot.
The two model tasks generate identical seeded observed datasets. The plot mode
checks their equality before making one PPC figure per observed dataset and a
separate shared-environment-contrast figure.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_model import ACE_PARAM_NAMES, COV_FEATURE_NAMES, RESULTS_DIR, VAR_REDUCTION, resolve
from ae_model import AELogitPrior, AEPosterior
from posterior_predictive_checks import posterior_predictive_replicates, summarize_ppc
from training_budget_utils import (
    resolve_device, seed_from, simulate_fixed_n,
    simulate_summaries_given_ace, train_transient_posterior,
)


CASES = (
    ("ace_truth", (0.3, 0.3, 0.4)),
    ("ae_truth", (0.6, 0.0, 0.4)),
)
MODEL_ORDER = ("ae", "ace")
MODEL_LABELS = {"ae": "AE NPE", "ace": "ACE NPE"}
MODEL_COLORS = {"ae": "#CC6677", "ace": "#4477AA"}


def expected_config(args: argparse.Namespace, model: str) -> dict:
    return {
        "step": 11,
        "model": model,
        "K": args.k,
        "N_pairs_per_group": args.n,
        "observed_truths": {name: list(theta) for name, theta in CASES},
        "prior": "Beta(1,1) on A; C=0; E=1-A" if model == "ae" else "Dirichlet(1,1,1) on (A,C,E)",
        "theta_coordinates": ["log(A/E)"] if model == "ae" else ["log(A/E)", "log(C/E)"],
        "feature_cols": list(COV_FEATURE_NAMES),
        "var_feature": VAR_REDUCTION,
        "n_posterior_draws": args.n_posterior_draws,
        "n_ppc_replicates": args.n_ppc_replicates,
        "seed": args.seed,
        "training_seed": args.training_seed,
        "training": {
            "validation_fraction": args.validation_fraction,
            "epochs": args.epochs,
            "stop_after_epochs": args.stop_after_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "flow_type": args.flow_type,
            "flow_hidden": args.flow_hidden,
            "flow_transforms": args.flow_transforms,
        },
    }


def simulate_observed(args: argparse.Namespace) -> np.ndarray:
    rows = []
    for case_index, (_, theta) in enumerate(CASES, start=1):
        rows.append(simulate_summaries_given_ace(
            np.asarray(theta, dtype=float)[None, :], args.n,
            seed_from(args.seed, case_index, 20),
        )[0])
    return np.asarray(rows, dtype=np.float32)


def train_ae(features: np.ndarray, ace: np.ndarray, args: argparse.Namespace):
    """Match Step 03/06 training settings with a one-coordinate AE target."""
    n_scaler_train = max(2, int((1.0 - args.validation_fraction) * len(features)))
    scaler = StandardScaler().fit(features[:n_scaler_train])
    x = scaler.transform(features).astype(np.float32)
    theta = np.log(ace[:, 0:1].astype(np.float64) / ace[:, 2:3]).astype(np.float32)

    prior = AELogitPrior(device=args.device_resolved)
    torch.manual_seed(args.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.training_seed)
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
        device=str(args.device_resolved),
    )
    inference.append_simulations(
        theta=torch.as_tensor(theta), x=torch.as_tensor(x)
    )
    estimator = inference.train(
        training_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_num_epochs=args.epochs,
        stop_after_epochs=args.stop_after_epochs,
        validation_fraction=args.validation_fraction,
        show_train_summary=True,
    )
    latent_posterior = inference.build_posterior(estimator)
    latent_posterior.to("cpu")
    return AEPosterior(latent_posterior), scaler


def run_model(args: argparse.Namespace, output_dir: Path) -> None:
    model = args.model
    model_dir = output_dir / "models" / model
    model_dir.mkdir(parents=True, exist_ok=True)
    sample_path = model_dir / "samples.npz"
    if sample_path.exists() and not args.overwrite:
        raise FileExistsError(f"{sample_path} already exists; pass --overwrite to replace it")

    training_data_seed = seed_from(args.seed, 11, 1 if model == "ae" else 2)
    print(f"Training {model.upper()} NPE: K={args.k:,}, N={args.n:,}", flush=True)
    if model == "ace":
        features, ace = simulate_fixed_n(args.k, args.n, training_data_seed)
        posterior, scaler = train_transient_posterior(features, ace, args)
    else:
        rng = np.random.default_rng(seed_from(training_data_seed, 1))
        a = rng.beta(1.0, 1.0, size=args.k).clip(1e-6, 1.0 - 1e-6)
        ace = np.column_stack((a, np.zeros(args.k), 1.0 - a)).astype(np.float32)
        features = simulate_summaries_given_ace(
            ace, args.n, seed_from(training_data_seed, 2)
        ).astype(np.float32)
        posterior, scaler = train_ae(features, ace, args)

    observed = simulate_observed(args)
    draws_by_case = []
    replicates_by_case = []
    for case_index, (case_name, _) in enumerate(CASES, start=1):
        torch.manual_seed(seed_from(args.seed, case_index, 100 if model == "ae" else 200))
        x = torch.as_tensor(scaler.transform(observed[case_index - 1:case_index]), dtype=torch.float32)
        with torch.no_grad():
            draws = posterior.sample(
                (args.n_posterior_draws,), x=x, show_progress_bars=False
            ).cpu().numpy()
        if draws.shape != (args.n_posterior_draws, 3) or not np.isfinite(draws).all():
            raise ValueError(f"Invalid posterior draws for {model}/{case_name}")
        if model == "ae" and not np.all(draws[:, 1] == 0):
            raise ValueError("AE posterior must set C=0 exactly")
        replicates = posterior_predictive_replicates(
            draws, args.n, args.n_ppc_replicates,
            seed_from(args.seed, case_index, 300 if model == "ae" else 400),
        )
        draws_by_case.append(draws.astype(np.float32))
        replicates_by_case.append(replicates.astype(np.float32))
        print(f"  {case_name}: posterior and PPC draws complete", flush=True)

    # Save the fitted model for later inspection, plus compact plot inputs.
    with open(model_dir / "posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)
    with open(model_dir / "scaler.pkl", "wb") as handle:
        pickle.dump(scaler, handle)
    with open(model_dir / "config.json", "w") as handle:
        json.dump(expected_config(args, model), handle, indent=2)
    with open(sample_path, "wb") as handle:
        np.savez_compressed(
            handle, observed=observed,
            truths=np.asarray([theta for _, theta in CASES], dtype=np.float32),
            posterior=np.stack(draws_by_case),
            replicates=np.stack(replicates_by_case),
        )
    print(f"Saved {model.upper()} model and PPC samples -> {model_dir}", flush=True)


def shared_environment_contrast(features: np.ndarray) -> np.ndarray:
    """T_C = DZ covariance - one half of MZ covariance; E[T_C] = C/2."""
    return np.asarray(features)[..., 3] - 0.5 * np.asarray(features)[..., 1]


def save_ppc_figure(case_index: int, samples: dict, output_dir: Path) -> None:
    case_name, truth = CASES[case_index]
    fig, axes = plt.subplots(2, 5, figsize=(19, 7.6), squeeze=False)
    observed = samples["ae"]["observed"][case_index]
    for col, feature in enumerate(COV_FEATURE_NAMES):
        combined = np.concatenate([
            samples[model]["replicates"][case_index, :, col]
            for model in MODEL_ORDER
        ] + [observed[col:col + 1]])
        bins = np.linspace(combined.min(), combined.max(), 31)
        for row, model in enumerate(MODEL_ORDER):
            ax = axes[row, col]
            values = samples[model]["replicates"][case_index, :, col]
            low, high = np.quantile(values, [0.025, 0.975])
            ax.hist(values, bins=bins, density=True, color=MODEL_COLORS[model], alpha=0.7)
            ax.axvspan(low, high, color=MODEL_COLORS[model], alpha=0.10)
            ax.axvline(observed[col], color="black", linestyle="--", linewidth=1.6)
            ax.set_xlabel(feature)
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nDensity")
            ax.grid(alpha=0.15)
        axes[1, col].sharex(axes[0, col])

    # Axes 4 in both rows show the joint covariance check.
    all_x = np.concatenate([
        samples[model]["replicates"][case_index, :, 1] for model in MODEL_ORDER
    ] + [observed[1:2]])
    all_y = np.concatenate([
        samples[model]["replicates"][case_index, :, 3] for model in MODEL_ORDER
    ] + [observed[3:4]])
    x_pad = max(0.01, 0.05 * np.ptp(all_x))
    y_pad = max(0.01, 0.05 * np.ptp(all_y))
    for row, model in enumerate(MODEL_ORDER):
        ax = axes[row, 4]
        rep = samples[model]["replicates"][case_index]
        ax.scatter(rep[:, 1], rep[:, 3], s=12, alpha=0.3, color=MODEL_COLORS[model])
        ax.plot(observed[1], observed[3], "*", color="black", markersize=13)
        ax.set(xlabel="mz_cov", ylabel="dz_cov",
               xlim=(all_x.min() - x_pad, all_x.max() + x_pad),
               ylim=(all_y.min() - y_pad, all_y.max() + y_pad))
        ax.grid(alpha=0.15)
    source_label = "ACE-generated" if case_name == "ace_truth" else "AE-generated"
    fig.suptitle(
        f"{source_label} observation: (A,C,E)={truth}; N={samples['N']:,} per group\n"
        "Dashed lines and stars: observed; shading: 95% posterior predictive interval",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_dir / f"ppc_{case_name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_contrast_figure(samples: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.6), sharex=True)
    for case_index, (case_name, truth) in enumerate(CASES):
        ax = axes[case_index]
        observed = float(shared_environment_contrast(
            samples["ae"]["observed"][case_index]
        ))
        contrast_by_model = {
            model: shared_environment_contrast(
                samples[model]["replicates"][case_index]
            ) for model in MODEL_ORDER
        }
        all_values = np.concatenate(list(contrast_by_model.values()) + [np.array([observed])])
        bins = np.linspace(all_values.min(), all_values.max(), 42)
        for model in MODEL_ORDER:
            ax.hist(contrast_by_model[model], bins=bins, density=True,
                    histtype="step", linewidth=2.0, color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model])
        ax.axvline(observed, color="black", linestyle="--", linewidth=1.7,
                   label="Observed")
        source_label = "ACE-generated" if case_name == "ace_truth" else "AE-generated"
        ax.set_ylabel(f"{source_label}\nDensity")
        ax.set_title(f"Generating C={truth[1]:.1f}; population T_C={truth[1] / 2:.2f}")
        ax.grid(alpha=0.15)
        ax.legend(frameon=False)
    axes[-1].set_xlabel(r"Shared-environment contrast $T_C = \mathrm{Cov}_{DZ} - \frac{1}{2}\mathrm{Cov}_{MZ}$")
    fig.tight_layout()
    fig.savefig(output_dir / "shared_environment_contrast.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_results(args: argparse.Namespace, output_dir: Path) -> None:
    samples = {}
    for model in MODEL_ORDER:
        model_dir = output_dir / "models" / model
        with open(model_dir / "config.json") as handle:
            config = json.load(handle)
        if config != expected_config(args, model):
            raise ValueError(f"{model_dir / 'config.json'} does not match these options")
        with np.load(model_dir / "samples.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        expected_shapes = {
            "observed": (2, 4), "truths": (2, 3),
            "posterior": (2, args.n_posterior_draws, 3),
            "replicates": (2, args.n_ppc_replicates, 4),
        }
        for key, shape in expected_shapes.items():
            if key not in data or data[key].shape != shape or not np.isfinite(data[key]).all():
                raise ValueError(f"Invalid {key} in {model_dir / 'samples.npz'}")
        samples[model] = data
    if not np.array_equal(samples["ae"]["observed"], samples["ace"]["observed"]):
        raise ValueError("AE and ACE tasks used different observed datasets")
    expected_truths = np.asarray([theta for _, theta in CASES], dtype=np.float32)
    if any(not np.array_equal(data["truths"], expected_truths) for data in samples.values()):
        raise ValueError("Saved generating parameters differ from Step 11 design")
    samples["N"] = args.n

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    ppc_rows = []
    contrast_rows = []
    for case_index, (case_name, truth) in enumerate(CASES):
        observed = samples["ae"]["observed"][case_index]
        observed_contrast = float(shared_environment_contrast(observed))
        for model in MODEL_ORDER:
            replicates = samples[model]["replicates"][case_index]
            for row in summarize_ppc(replicates, observed):
                ppc_rows.append({"dataset": case_name, "model": model, **row})
            values = shared_environment_contrast(replicates)
            low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
            contrast_rows.append({
                "dataset": case_name, "model": model, "true_C": truth[1],
                "population_T_C": truth[1] / 2, "observed_T_C": observed_contrast,
                "predictive_mean": float(values.mean()),
                "predictive_median": float(median),
                "predictive_q2_5": float(low), "predictive_q97_5": float(high),
                "upper_tail_fraction": float(np.mean(values >= observed_contrast)),
                "observed_in_95_interval": int(low <= observed_contrast <= high),
                "n_replicates": len(values),
            })
        save_ppc_figure(case_index, samples, figure_dir)
    save_contrast_figure(samples, figure_dir)
    pd.DataFrame(ppc_rows).to_csv(output_dir / "ppc_summary.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(output_dir / "contrast_summary.csv", index=False)
    print(f"Saved three figures and two summary tables -> {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", choices=MODEL_ORDER, help="Train and evaluate one model")
    mode.add_argument("--plot", action="store_true", help="Combine two completed model tasks")
    parser.add_argument("--k", type=int, default=200_000)
    parser.add_argument("--n", type=int, default=1_000)
    parser.add_argument("--n-posterior-draws", type=int, default=2_000)
    parser.add_argument("--n-ppc-replicates", type=int, default=500)
    parser.add_argument("--output-dir", default="step11_ae_ace_ppc")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--stop-after-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--flow-type", choices=("nsf", "maf", "maf_rqs", "mdn"), default="nsf")
    parser.add_argument("--flow-hidden", type=int, default=64)
    parser.add_argument("--flow-transforms", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.k < 100 or args.n < 2 or args.n_posterior_draws < 20:
        parser.error("K >= 100, N >= 2, and at least 20 posterior draws are required")
    if not 2 <= args.n_ppc_replicates <= args.n_posterior_draws:
        parser.error("PPC replicates must be between 2 and the posterior draw count")
    if not 0 < args.validation_fraction < 0.5:
        parser.error("validation fraction must be between 0 and 0.5")
    if min(args.epochs, args.stop_after_epochs, args.batch_size) < 1:
        parser.error("training counts must be positive")
    args.device_resolved = resolve_device(args.device)
    return args


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir, RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot_results(args, output_dir)
    else:
        run_model(args, output_dir)


if __name__ == "__main__":
    main()
