"""STEP 02b -- train one ACE NPE for each simulation scheme and fixed N.

The fixed-sum arms are trained in two additive-log-ratio coordinates,
``log(A/E)`` and ``log(C/E)``.  Saved posteriors transparently convert samples
back to three positive ACE components summing to V.

N is never supplied as a feature: each model is trained on exactly one fixed
sample size.  With the defaults this script trains 3 x 6 = 18 models.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_model import ACE_PARAM_NAMES, COV_FEATURE_NAMES, DATA_DIR, MODELS_DIR, VAR_REDUCTION, map_from_samples, resolve
from ace_prior_comparison import (
    ACEPosterior,
    FIXED_SUM_SCHEMES,
    SCHEMES,
    SCHEME_LABELS,
    SimplexALRPrior,
    DEFAULT_N_PAIRS,
    ace_to_latent,
    model_dir_for,
)

warnings.filterwarnings("ignore")


def load_manifest(data_dir: Path) -> dict:
    path = data_dir / "simulation_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run 01b_generate_prior_comparison_data.py first")
    with open(path) as f:
        return json.load(f)


def clean_data(df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    required = list(COV_FEATURE_NAMES) + list(ACE_PARAM_NAMES)
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"{scheme}: missing columns {missing}")
    if "simulation_scheme" in df and not (df["simulation_scheme"] == scheme).all():
        raise ValueError(f"{scheme}: CSV contains rows from another simulation scheme")
    cleaned = df.dropna(subset=required).copy()
    invalid = (cleaned["mz_cov"].abs() > cleaned["mz_var"].abs()) | (
        cleaned["dz_cov"].abs() > cleaned["dz_var"].abs()
    )
    if invalid.any():
        print(f"  Removing {int(invalid.sum())} invalid covariance rows")
        cleaned = cleaned.loc[~invalid].copy()
    if len(cleaned) < 20:
        raise ValueError(f"{scheme}: fewer than 20 usable rows")
    return cleaned


def evaluate(posterior, x_test, y_test, n_draws: int, n_eval: int):
    n_eval = min(n_eval, len(x_test))
    means, stds, maps, lows, highs = [], [], [], [], []
    for row in x_test[:n_eval]:
        x_obs = torch.as_tensor(row, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            samples = posterior.sample((n_draws,), x=x_obs, show_progress_bars=False)
        draws = samples.cpu().numpy()
        means.append(draws.mean(axis=0))
        stds.append(draws.std(axis=0))
        maps.append(map_from_samples(draws))
        lows.append(np.percentile(draws, 2.5, axis=0))
        highs.append(np.percentile(draws, 97.5, axis=0))
    return {
        "true": np.asarray(y_test[:n_eval]),
        "mean": np.asarray(means),
        "std": np.asarray(stds),
        "map": np.asarray(maps),
        "ci_lo": np.asarray(lows),
        "ci_hi": np.asarray(highs),
    }


def save_evaluation(evaluation: dict, scheme: str, output_dir: Path):
    rows = []
    true = evaluation["true"]
    pred = evaluation["mean"]
    for index, parameter in enumerate(ACE_PARAM_NAMES):
        error = pred[:, index] - true[:, index]
        rows.append(
            {
                "scheme": scheme,
                "parameter": parameter,
                "bias": float(error.mean()),
                "mae": float(mean_absolute_error(true[:, index], pred[:, index])),
                "rmse": float(np.sqrt(mean_squared_error(true[:, index], pred[:, index]))),
                "r2": float(r2_score(true[:, index], pred[:, index])),
                "mean_posterior_sd": float(evaluation["std"][:, index].mean()),
                "coverage_95": float(
                    np.mean(
                        (true[:, index] >= evaluation["ci_lo"][:, index])
                        & (true[:, index] <= evaluation["ci_hi"][:, index])
                    )
                ),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "test_metrics.csv", index=False)
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(rows, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, (axis, parameter) in enumerate(zip(axes, ACE_PARAM_NAMES)):
        axis.errorbar(
            true[:, i],
            pred[:, i],
            yerr=evaluation["std"][:, i],
            fmt="o",
            markersize=3,
            alpha=0.35,
            elinewidth=0.5,
        )
        lo = min(true[:, i].min(), pred[:, i].min())
        hi = max(true[:, i].max(), pred[:, i].max())
        axis.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
        axis.set(title=parameter, xlabel="True", ylabel="Posterior mean")
        axis.grid(alpha=0.25)
    fig.suptitle(f"Held-out performance: {SCHEME_LABELS[scheme]}")
    fig.tight_layout()
    fig.savefig(output_dir / "predictions_vs_true.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 4))
    values = metrics["mean_posterior_sd"].to_numpy()
    bars = axis.bar(ACE_PARAM_NAMES, values, color="steelblue")
    axis.bar_label(bars, fmt="%.4f")
    axis.set(ylabel="Mean posterior SD", title=f"Posterior uncertainty: {SCHEME_LABELS[scheme]}")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "posterior_uncertainty.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def train_one(
    args,
    scheme: str,
    n_pairs: int,
    data_dir: Path,
    models_dir: Path,
    manifest: dict,
):
    data_file = manifest.get("files", {}).get(scheme, {}).get(
        str(n_pairs), f"ace_{scheme}_N{n_pairs}.csv"
    )
    data_path = data_dir / data_file
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")
    output_dir = model_dir_for(models_dir, scheme, n_pairs)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print(f"TRAINING {SCHEME_LABELS[scheme]} | FIXED N={n_pairs}")
    print("=" * 78)
    print(f"Data:   {data_path}")
    print(f"Output: {output_dir}")

    df = clean_data(pd.read_csv(data_path), scheme)
    if "N_pairs" not in df:
        raise ValueError(f"{scheme}, N={n_pairs}: CSV has no N_pairs column")
    observed_n = set(df["N_pairs"].astype(int).unique())
    if observed_n != {n_pairs}:
        raise ValueError(
            f"{scheme}, N={n_pairs}: expected one fixed N but found {sorted(observed_n)}"
        )
    feature_cols = list(COV_FEATURE_NAMES)

    x = df[feature_cols].to_numpy(dtype=np.float32)
    ace = df[ACE_PARAM_NAMES].to_numpy(dtype=np.float32)
    total_variance = float(manifest.get("total_variance_fixed_sum", 1.0))
    alpha = manifest.get("dirichlet_alpha", [1.0, 1.0, 1.0])
    theta = ace_to_latent(ace, scheme)

    rng = np.random.default_rng(args.split_seed)
    indices = rng.permutation(len(df))
    n_train = int(0.70 * len(df))
    n_val = int(0.15 * len(df))
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    scaler = StandardScaler().fit(x[train_idx])
    x_train = scaler.transform(x[train_idx]).astype(np.float32)
    x_val = scaler.transform(x[val_idx]).astype(np.float32)
    x_test = scaler.transform(x[test_idx]).astype(np.float32)
    joblib.dump(scaler, output_dir / "feature_scaler.pkl")

    if scheme == "independent_uniform":
        y_min = ace[train_idx].min(axis=0)
        y_max = ace[train_idx].max(axis=0)
        buffer = args.prior_buffer * (y_max - y_min)
        prior_lower = torch.as_tensor(y_min - buffer, dtype=torch.float32)
        prior_upper = torch.as_tensor(y_max + buffer, dtype=torch.float32)
        prior = BoxUniform(
            low=prior_lower,
            high=prior_upper,
            device=str(args.device_resolved),
        )
        parameterization = "ACE directly"
        latent_names = list(ACE_PARAM_NAMES)
    else:
        prior = SimplexALRPrior(scheme, alpha, device=args.device_resolved)
        parameterization = "ALR: log(A/E), log(C/E); E reconstructed"
        latent_names = ["log_A_over_E", "log_C_over_E"]

    torch.manual_seed(args.training_seed)
    density_builder = posterior_nn(
        model=args.flow_type,
        embedding_net=nn.Identity(),
        hidden_features=args.flow_hidden,
        num_transforms=args.flow_transforms,
        z_score_theta="independent",
        z_score_x="independent",
    )
    inference = SNPE(prior=prior, density_estimator=density_builder, device=str(args.device_resolved))
    theta_all = np.vstack([theta[train_idx], theta[val_idx]])
    x_all = np.vstack([x_train, x_val])
    inference.append_simulations(
        theta=torch.as_tensor(theta_all, dtype=torch.float32),
        x=torch.as_tensor(x_all, dtype=torch.float32),
    )
    estimator = inference.train(
        training_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_num_epochs=args.epochs,
        stop_after_epochs=args.stop_after_epochs,
        validation_fraction=len(val_idx) / len(theta_all),
        show_train_summary=True,
    )
    latent_posterior = inference.build_posterior(estimator)
    latent_posterior.to("cpu")
    posterior = ACEPosterior(latent_posterior, scheme, total_variance)

    with open(output_dir / "posterior.pkl", "wb") as f:
        pickle.dump(posterior, f)
    torch.save(
        {
            "density_estimator_state_dict": estimator.state_dict(),
            "scheme": scheme,
            "theta_parameterization": parameterization,
            **(
                {
                    "prior_lower": prior_lower,
                    "prior_upper": prior_upper,
                }
                if scheme == "independent_uniform"
                else {}
            ),
        },
        output_dir / "density_estimator.pt",
    )

    config = {
        "model_type": "NPE_prior_comparison",
        "simulation_scheme": scheme,
        "simulation_label": SCHEME_LABELS[scheme],
        "fixed_n_pairs": n_pairs,
        "n_is_model_feature": False,
        "theta_parameterization": parameterization,
        "latent_param_names": latent_names,
        "param_names": list(ACE_PARAM_NAMES),
        "total_variance": total_variance if scheme in FIXED_SUM_SCHEMES else None,
        "dirichlet_alpha": alpha if scheme == "dirichlet" else None,
        "prior_type": (
            "boxuniform_from_training_range"
            if scheme == "independent_uniform"
            else f"exact_{scheme}_ALR"
        ),
        **(
            {
                "prior_buffer": args.prior_buffer,
                "prior_lower": prior_lower.tolist(),
                "prior_upper": prior_upper.tolist(),
            }
            if scheme == "independent_uniform"
            else {}
        ),
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "var_feature": VAR_REDUCTION,
        "flow_type": args.flow_type,
        "flow_hidden": args.flow_hidden,
        "flow_transforms": args.flow_transforms,
        "training_rows": len(train_idx),
        "validation_rows": len(val_idx),
        "test_rows": len(test_idx),
        "source_data": str(data_path),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    evaluation = evaluate(
        posterior,
        x_test,
        ace[test_idx],
        n_draws=args.n_posterior_samples,
        n_eval=args.n_eval,
    )
    save_evaluation(evaluation, scheme, output_dir)
    print(f"Completed {scheme}, N={n_pairs} -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train one ACE NPE per simulation scheme and fixed sample size"
    )
    parser.add_argument("--data_dir", default="prior_comparison")
    parser.add_argument("--output_dir", default="prior_comparison")
    parser.add_argument("--schemes", nargs="+", choices=SCHEMES, default=list(SCHEMES))
    parser.add_argument(
        "--n_pairs",
        type=int,
        nargs="+",
        default=None,
        help="Fixed-N models to train (default: 50 100 500 1000 5000 20000)",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--stop_after_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--prior_buffer",
        type=float,
        default=0.5,
        help="Independent-arm prior buffer, matching STEP 02 (default: 0.5)",
    )
    parser.add_argument("--flow_type", choices=["nsf", "maf", "maf_rqs", "mdn"], default="nsf")
    parser.add_argument("--flow_hidden", type=int, default=64)
    parser.add_argument("--flow_transforms", type=int, default=5)
    parser.add_argument("--n_posterior_samples", type=int, default=500)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--training_seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.n_posterior_samples <= 1 or args.n_eval <= 0:
        parser.error("epochs, batch_size, and n_eval must be positive; posterior samples must exceed 1")
    if args.prior_buffer < 0:
        parser.error("--prior_buffer must be non-negative")
    if args.device == "auto":
        if torch.backends.mps.is_available():
            args.device_resolved = torch.device("mps")
        elif torch.cuda.is_available():
            args.device_resolved = torch.device("cuda")
        else:
            args.device_resolved = torch.device("cpu")
    else:
        args.device_resolved = torch.device(args.device)

    data_dir = resolve(args.data_dir, DATA_DIR)
    models_dir = resolve(args.output_dir, MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(data_dir)
    n_values = tuple(args.n_pairs) if args.n_pairs is not None else tuple(
        manifest.get("fixed_n_pairs", DEFAULT_N_PAIRS)
    )
    if any(n < 2 for n in n_values):
        parser.error("--n_pairs values must be at least 2")
    print(f"Training device: {args.device_resolved}")
    for scheme in args.schemes:
        for n_pairs in n_values:
            train_one(args, scheme, n_pairs, data_dir, models_dir, manifest)


if __name__ == "__main__":
    main()
