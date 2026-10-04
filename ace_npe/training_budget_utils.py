"""Shared fixed-N simulation, transient training, and evaluation utilities.

STEP 03 and STEP 06 intentionally use these same functions so their training
and diagnostic results differ only in the requested K grid and output plots.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn

from ace_model import ACE_PARAM_NAMES
from prior_compare.prior_compare_utils import (
    ACEPosterior,
    SimplexALRPrior,
    ace_to_latent,
)


def seed_from(base_seed: int, *keys: int) -> int:
    """Create a reproducible 32-bit seed from a base seed and integer keys."""
    state = np.random.SeedSequence([base_seed, *keys]).generate_state(1)
    return int(state[0])


def _wishart_compound_symmetric_summaries(
    correlations: np.ndarray,
    n_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw exact Gaussian sample-covariance summaries without raw twin data."""
    correlations = np.asarray(correlations, dtype=np.float64)
    if correlations.ndim != 1:
        raise ValueError("correlations must be one-dimensional")
    if n_pairs < 2:
        raise ValueError("n_pairs must be at least 2")
    if np.any(np.abs(correlations) >= 1.0):
        raise ValueError("correlations must lie strictly between -1 and 1")

    degrees_freedom = n_pairs - 1
    a11_rng = np.random.default_rng(seed_from(seed, 1))
    a21_rng = np.random.default_rng(seed_from(seed, 2))
    a22_rng = np.random.default_rng(seed_from(seed, 3))
    a11 = np.sqrt(a11_rng.chisquare(degrees_freedom, size=len(correlations)))
    a21 = a21_rng.standard_normal(size=len(correlations))
    a22 = np.sqrt(
        a22_rng.chisquare(degrees_freedom - 1, size=len(correlations))
    )

    l22 = np.sqrt(np.maximum(1.0 - correlations**2, 0.0))
    b11 = a11
    b21 = correlations * a11 + l22 * a21
    b22 = l22 * a22
    s11 = b11**2 / degrees_freedom
    s12 = b11 * b21 / degrees_freedom
    s22 = (b21**2 + b22**2) / degrees_freedom
    return s11, s12, s22


def simulate_fixed_n(
    n_simulations: int,
    n_pairs: int,
    seed: int,
    return_covariances: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return four covariance features and Dirichlet(1,1,1) ACE truths."""
    if n_simulations < 2:
        raise ValueError("n_simulations must be at least 2")

    theta_rng = np.random.default_rng(seed_from(seed, 1))
    ace = theta_rng.dirichlet(np.ones(3), size=n_simulations)
    mz_correlation = ace[:, 0] + ace[:, 1]
    dz_correlation = 0.5 * ace[:, 0] + ace[:, 1]
    mz_var1, mz_cov, mz_var2 = _wishart_compound_symmetric_summaries(
        mz_correlation, n_pairs, seed_from(seed, 2)
    )
    dz_var1, dz_cov, dz_var2 = _wishart_compound_symmetric_summaries(
        dz_correlation, n_pairs, seed_from(seed, 3)
    )
    mz_var = 0.5 * (mz_var1 + mz_var2)
    dz_var = 0.5 * (dz_var1 + dz_var2)
    features = np.column_stack((mz_var, mz_cov, dz_var, dz_cov)).astype(
        np.float32
    )
    ace = ace.astype(np.float32)
    if not return_covariances:
        return features, ace
    covariance_data = pd.DataFrame(
        {
            "mz_var1": mz_var1,
            "mz_cov": mz_cov,
            "mz_var2": mz_var2,
            "dz_var1": dz_var1,
            "dz_cov": dz_cov,
            "dz_var2": dz_var2,
            "mz_var": mz_var,
            "dz_var": dz_var,
        }
    )
    return features, ace, covariance_data


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def train_transient_posterior(
    features: np.ndarray,
    ace: np.ndarray,
    args: argparse.Namespace,
) -> tuple[ACEPosterior, StandardScaler]:
    """Train one in-memory Dirichlet NPE; do not serialize the fitted model."""
    n_rows = len(features)
    n_scaler_train = max(2, int((1.0 - args.validation_fraction) * n_rows))
    scaler = StandardScaler().fit(features[:n_scaler_train])
    scaled = scaler.transform(features).astype(np.float32)
    theta = ace_to_latent(ace, "dirichlet")

    prior = SimplexALRPrior(
        "dirichlet", (1.0, 1.0, 1.0), device=args.device_resolved
    )
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
        theta=torch.as_tensor(theta, dtype=torch.float32),
        x=torch.as_tensor(scaled, dtype=torch.float32),
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
    return ACEPosterior(latent_posterior, "dirichlet", 1.0), scaler


def evaluate_cell(
    posterior: ACEPosterior,
    scaler: StandardScaler,
    test_features: np.ndarray,
    test_ace: np.ndarray,
    k_value: int,
    n_pairs: int,
    n_draws: int,
    posterior_seed: int,
    *,
    draw_callback: Callable[[int, np.ndarray], None] | None = None,
) -> pd.DataFrame:
    """Create one compact row per test dataset and ACE parameter.

    An optional callback receives the one-based dataset index and the same
    joint draws used for marginal summaries. It must not mutate the draws or
    change the global RNG state.
    """
    torch.manual_seed(posterior_seed)
    scaled = scaler.transform(test_features).astype(np.float32)
    rows: list[dict[str, float | int | str]] = []
    for test_index, (feature_row, truth_row) in enumerate(
        zip(scaled, test_ace), start=1
    ):
        x = torch.as_tensor(feature_row, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            draws = posterior.sample(
                (n_draws,), x=x, show_progress_bars=False
            ).cpu().numpy()
        if draw_callback is not None:
            draw_callback(test_index, draws)
        for parameter_index, parameter in enumerate(ACE_PARAM_NAMES):
            values = draws[:, parameter_index]
            truth = float(truth_row[parameter_index])
            posterior_mean = float(values.mean())
            error = posterior_mean - truth
            rows.append(
                {
                    "K": k_value,
                    "N": n_pairs,
                    "model_index_h": 1,
                    "test_dataset_m": test_index,
                    "parameter": parameter,
                    "truth": truth,
                    "posterior_mean": posterior_mean,
                    "error": error,
                    "absolute_error": abs(error),
                    "posterior_sd": float(values.std(ddof=1)),
                    "posterior_skewness": float(skew(values, bias=False)),
                    "posterior_kurtosis": float(
                        kurtosis(values, fisher=False, bias=False)
                    ),
                    "ci_2_5": float(np.percentile(values, 2.5)),
                    "ci_97_5": float(np.percentile(values, 97.5)),
                    "true_rank": int(np.sum(values < truth)),
                    "true_rank_percentile": float(np.mean(values < truth)),
                }
            )
    return pd.DataFrame(rows)
