"""Likelihood-free SBC quantities for the fixed-total-variance ACE model."""

from __future__ import annotations

import numpy as np


RMSE_SBC_DEFINITION = {
    "quantity": "covariance_prediction_rmse",
    "features": ["mz_var", "mz_cov", "dz_var", "dz_cov"],
    "prediction": ["1", "A+C", "1", "0.5*A+C"],
    "total_variance": 1.0,
    "scale": "original covariance units; equal weights",
    "formula": "sqrt(mean((prediction(theta) - observed_features)**2))",
    "rank": "number strictly below truth plus uniform integer in [0, n_ties]",
    "posterior_draws": "same joint draws as marginal SBC",
}


def covariance_prediction_rmse(
    ace: np.ndarray, observed_features: np.ndarray
) -> np.ndarray:
    """Evaluate each theta against one fixed, unstandardized observation.

    ACE's total variance is fixed at one in this experiment. Use that exact
    value rather than the rounded sum of float32 posterior components.
    No noisy replicate data, posterior means, or likelihood are used.
    """
    ace = np.asarray(ace, dtype=np.float64)
    observed = np.asarray(observed_features, dtype=np.float64)
    if ace.ndim < 1 or ace.shape[-1] != 3:
        raise ValueError("ACE parameters must have final dimension 3")
    if observed.shape != (4,):
        raise ValueError("observed_features must have shape (4,)")
    if not np.all(np.isfinite(ace)) or not np.all(np.isfinite(observed)):
        raise ValueError("RMSE inputs must be finite")
    if np.any(ace < 0) or not np.allclose(
        ace.sum(axis=-1), 1.0, rtol=0.0, atol=1e-6
    ):
        raise ValueError("This quantity requires nonnegative ACE values summing to 1")
    predicted = np.empty(ace.shape[:-1] + (4,), dtype=np.float64)
    predicted[..., 0] = 1.0
    predicted[..., 1] = ace[..., 0] + ace[..., 1]
    predicted[..., 2] = 1.0
    predicted[..., 3] = 0.5 * ace[..., 0] + ace[..., 1]
    return np.sqrt(np.mean(np.square(predicted - observed), axis=-1))


def summarize_rmse_sbc(
    draws: np.ndarray,
    truth: np.ndarray,
    observed_features: np.ndarray,
    tie_rng: np.random.Generator,
) -> dict[str, float | int]:
    """Rank the generating theta's RMSE among joint posterior-draw RMSEs.

    Exact ties are randomized using a separate RNG so parameter ranks and
    posterior sampling are unaffected. Retain compact summaries only.
    """
    draws = np.asarray(draws)
    if draws.ndim != 2 or draws.shape[1] != 3 or len(draws) < 2:
        raise ValueError("draws must have shape (L, 3), with L >= 2")
    if np.asarray(truth).shape != (3,):
        raise ValueError("truth must have shape (3,)")
    true_rmse = float(covariance_prediction_rmse(truth, observed_features))
    posterior_rmse = covariance_prediction_rmse(draws, observed_features)
    strict_rank = int(np.count_nonzero(posterior_rmse < true_rmse))
    n_ties = int(np.count_nonzero(posterior_rmse == true_rmse))
    rank = strict_rank + (int(tie_rng.integers(n_ties + 1)) if n_ties else 0)
    return {
        "true_rmse": true_rmse,
        "posterior_rmse_mean": float(posterior_rmse.mean()),
        "posterior_rmse_sd": float(posterior_rmse.std(ddof=1)),
        "posterior_rmse_median": float(np.median(posterior_rmse)),
        "posterior_rmse_q2_5": float(np.percentile(posterior_rmse, 2.5)),
        "posterior_rmse_q97_5": float(np.percentile(posterior_rmse, 97.5)),
        "rank_strict": strict_rank,
        "rank_ties": n_ties,
        "true_rank": rank,
        "true_rank_percentile": rank / len(draws),
        "n_posterior_draws": len(draws),
    }
