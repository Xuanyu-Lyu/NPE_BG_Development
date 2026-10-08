"""SBC log-gamma scores from retained integer ranks, without posterior draws.

Uses the Modrak et al. (2025), equation 7, statistic implemented by BayesFlow
2.0.14. Null ranks are uniform on 0,...,L. The full L+1 rank grid is used,
independently of the existing plotting-band grid. Thresholds apply to one
test quantity and are not adjusted across parameters or experiment cells.
"""

from functools import lru_cache

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom


LOG_GAMMA_SETTINGS = {
    "version": 1,
    "formula": "log(gamma / gamma_threshold)",
    "statistic": "2 * min_j min(P(B_j <= R_j), P(B_j >= R_j))",
    "rank_grid": "j=1,...,L+1; z_j=j/(L+1); R_j=count(rank<j)",
    "null": "M independent ranks uniform on integers 0,...,L",
    "num_null_draws": 1000,
    "quantile": 0.05,
    "seed": 20261007,
    "threshold_scope": "one quantity; no adjustment across quantities or cells",
    "source": "https://bayesflow.org/v2.0.14/_modules/bayesflow/diagnostics/metrics/calibration_log_gamma.html",
}


def _log_gamma_from_counts(counts: np.ndarray, n_tests: int, grid: np.ndarray) -> float:
    """Compute in log space, including an exact-PMF fallback for underflow."""
    lower = binom.logcdf(counts, n_tests, grid)
    upper = binom.logsf(counts - 1, n_tests, grid)
    # SciPy's logcdf/logsf can themselves underflow for extreme nonuniformity.
    # Sum log PMFs only at affected points; no clipping of the final score.
    for index in np.flatnonzero(np.isneginf(lower)):
        lower[index] = logsumexp(binom.logpmf(
            np.arange(counts[index] + 1), n_tests, grid[index]
        ))
    for index in np.flatnonzero(np.isneginf(upper)):
        upper[index] = logsumexp(binom.logpmf(
            np.arange(counts[index], n_tests + 1), n_tests, grid[index]
        ))
    return float(np.log(2.0) + np.min(np.minimum(lower, upper)))


def log_gamma_discrepancy(ranks: np.ndarray, n_draws: int) -> float:
    """Natural log of raw gamma; reuse any tie-randomized ranks unchanged."""
    ranks = np.asarray(ranks, dtype=float)
    if not isinstance(n_draws, (int, np.integer)) or n_draws < 1:
        raise ValueError("n_draws must be a positive integer")
    if (ranks.ndim != 1 or len(ranks) < 2 or not np.isfinite(ranks).all()
            or np.any(ranks != np.floor(ranks))
            or np.any((ranks < 0) | (ranks > n_draws))):
        raise ValueError("ranks must be a finite integer vector in [0, L], with M >= 2")
    thresholds = np.arange(1, n_draws + 2)
    counts = np.searchsorted(np.sort(ranks), thresholds, side="left")
    return _log_gamma_from_counts(counts, len(ranks), thresholds / (n_draws + 1))


def gamma_null_distribution(
    n_tests: int, n_draws: int, num_null_draws: int, seed: int
) -> np.ndarray:
    """Simulate BayesFlow's null ECDF counts using a local seeded generator.

    Counting U(0,1) values below j/(L+1) is equivalent to counting iid discrete
    uniform ranks below j. Sorting avoids an M by (L+1) comparison matrix.
    """
    if n_tests < 2 or n_draws < 1 or num_null_draws < 1:
        raise ValueError("Require M >= 2, L >= 1, and a positive null simulation count")
    rng = np.random.default_rng(seed)
    grid = np.arange(1, n_draws + 2) / (n_draws + 1)
    values = np.empty(num_null_draws)
    for index in range(num_null_draws):
        counts = np.searchsorted(np.sort(rng.random(n_tests)), grid, side="left")
        values[index] = np.exp(_log_gamma_from_counts(counts, n_tests, grid))
    return values


@lru_cache(maxsize=32)
def gamma_null_threshold(
    n_tests: int, n_draws: int, num_null_draws: int, quantile: float, seed: int
) -> float:
    """Reuse one deterministic reference for all quantities with matching M/L."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie strictly between 0 and 1")
    threshold = float(np.quantile(
        gamma_null_distribution(n_tests, n_draws, num_null_draws, seed), quantile
    ))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("The simulated gamma threshold must be finite and positive")
    return threshold


def summarize_log_gamma(ranks: np.ndarray, n_draws: int) -> dict:
    """Return a reproducible score and its null-reference settings."""
    log_gamma = log_gamma_discrepancy(ranks, n_draws)
    settings = LOG_GAMMA_SETTINGS
    threshold = gamma_null_threshold(
        len(ranks), n_draws, settings["num_null_draws"], settings["quantile"], settings["seed"]
    )
    score = log_gamma - np.log(threshold)
    return {
        "M": len(ranks), "L": n_draws,
        "gamma": float(np.exp(log_gamma)),
        "gamma_threshold": threshold,
        "log_gamma": float(score),
        "rejects_uniformity": bool(score < 0),
        "null_quantile": settings["quantile"],
        "num_null_draws": settings["num_null_draws"],
        "null_seed": settings["seed"],
    }
