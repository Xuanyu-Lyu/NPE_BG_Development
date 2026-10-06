"""Posterior predictive checks in original covariance-summary units.

These are descriptive model-fit checks. Tail fractions and interval inclusion
are not SBC ranks or calibrated hypothesis-test p-values.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ace_model import COV_FEATURE_NAMES
from training_budget_utils import seed_from, simulate_summaries_given_ace


PPC_DEFINITION = {
    "version": 1,
    "features": list(COV_FEATURE_NAMES),
    "simulator": "Gaussian sample covariance via Wishart; df=N-1; total variance=1",
    "parameters": "joint posterior draws selected without replacement",
    "replicates": "one independent noisy MZ/DZ replicate per selected draw, at observed N",
    "scale": "original covariance units",
    "interval": "equal-tailed 95% posterior predictive interval",
    "tail_fraction": "fraction of replicated summaries >= observed summary; descriptive only",
}


def posterior_predictive_replicates(
    draws: np.ndarray, n_pairs: int, n_replicates: int, seed: int
) -> np.ndarray:
    """Select existing joint draws with a local RNG, then simulate fresh data."""
    if not 2 <= n_replicates <= len(draws):
        raise ValueError("n_replicates must be between 2 and the number of posterior draws")
    rng = np.random.default_rng(seed_from(seed, 1))
    indices = rng.choice(len(draws), size=n_replicates, replace=False)
    return simulate_summaries_given_ace(draws[indices], n_pairs, seed_from(seed, 2))


def summarize_ppc(replicates: np.ndarray, observed: np.ndarray) -> list[dict]:
    """One compact row per observed summary; no generating truth is required."""
    replicates = np.asarray(replicates, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if (replicates.ndim != 2 or replicates.shape[1] != 4 or len(replicates) < 2
            or observed.shape != (4,) or not np.isfinite(replicates).all()
            or not np.isfinite(observed).all()):
        raise ValueError("PPC requires finite replicates of shape (R, 4) and observations (4,)")
    rows = []
    for index, feature in enumerate(COV_FEATURE_NAMES):
        values = replicates[:, index]
        low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
        rows.append({
            "feature": feature,
            "observed": float(observed[index]),
            "predictive_mean": float(values.mean()),
            "predictive_sd": float(values.std(ddof=1)),
            "predictive_q2_5": float(low),
            "predictive_median": float(median),
            "predictive_q97_5": float(high),
            "upper_tail_fraction": float(np.mean(values >= observed[index])),
            "observed_in_95_interval": int(low <= observed[index] <= high),
            "n_replicates": len(values),
        })
    return rows


def save_ppc_plots(replicates: np.ndarray, observed: np.ndarray, path: Path, title: str) -> None:
    """Four marginal PPCs and a paired MZ/DZ covariance view for one dataset."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for index, feature in enumerate(COV_FEATURE_NAMES):
        ax = axes.flat[index]
        values = replicates[:, index]
        low, high = np.quantile(values, [0.025, 0.975])
        ax.hist(values, bins=30, density=True, color="#4477AA", alpha=0.75,
                label="Replicated")
        ax.axvspan(low, high, color="#4477AA", alpha=0.12)
        ax.axvline(observed[index], color="#CC3311", linestyle="--", label="Observed")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        if index == 0:
            ax.legend(frameon=False)
    ax = axes.flat[4]
    ax.scatter(replicates[:, 1], replicates[:, 3], s=12, alpha=0.3, color="#4477AA")
    ax.plot(observed[1], observed[3], "*", color="#CC3311", markersize=13,
            label="Observed")
    ax.set_xlabel("mz_cov")
    ax.set_ylabel("dz_cov")
    ax.legend(frameon=False)
    axes.flat[5].set_visible(False)
    fig.suptitle(title + f"\n{len(replicates)} noisy replicates; shaded intervals: 95%", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
