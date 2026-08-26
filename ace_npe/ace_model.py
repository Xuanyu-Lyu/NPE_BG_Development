"""
Shared library for the ACE / NPE pipeline.

Everything that more than one pipeline step needs lives here, because the
pipeline scripts themselves are numbered (``01_``, ``02_``, …) and a module
whose name starts with a digit cannot be imported with a normal ``import``
statement.  Numbered scripts therefore import from this module and never
from each other.

Contents
--------
Paths
    SCRIPT_DIR, DATA_DIR, RESULTS_DIR, MODELS_DIR, SIM_DIR, ANALYSIS_DIR,
    SE_CAL_DIR, DEMO_DIR — the single source of truth for where things are
    read/written.

ACE model math
    ``simulate_covariances``  — draw sample MZ/DZ covariance matrices.
    ``theoretical_covariances`` — the noise-free expected matrices.
    ``generate_training_data``  — full (θ, x) training-set simulator.

NPE helpers
    ``ACEEmbeddingNet``  — embedding network (see note below).
    ``map_from_samples`` — per-parameter MAP via 1-D KDE.
    ``load_posterior``   — load a trained run and report its feature layout.
    ``build_features``   — assemble a feature vector matching a run's config.
    ``posterior_stats``  — mean / SD / MAP / credible interval for one obs.

Note on ACEEmbeddingNet
-----------------------
This class is defined and exported, but ``02_train_npe.py`` currently passes
``nn.Identity()`` to the flow instead, so the flow conditions directly on the
4–5 standardized summary statistics.  See the "Why no embedding network?"
section of README.md.  The class is kept because (a) two archived posteriors
pickle-reference it by name and (b) it is the drop-in replacement if the
feature vector ever grows beyond a handful of summary statistics.
"""

from pathlib import Path

import json
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from scipy.stats import gaussian_kde


# ============================================================================
# PATHS  — single source of truth for the pipeline's layout
# ============================================================================

SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_DIR     = SCRIPT_DIR / "data"
RESULTS_DIR  = SCRIPT_DIR / "results"
MODELS_DIR   = RESULTS_DIR / "models"
SIM_DIR      = RESULTS_DIR / "simulations"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
SE_CAL_DIR   = RESULTS_DIR / "se_calibration"
DEMO_DIR     = RESULTS_DIR / "demo"

# Default trained run used by the demo and the posterior-recovery study.
DEFAULT_MODEL_DIR = MODELS_DIR / "se_proxy"


def resolve(path, base: Path) -> Path:
    """Resolve *path* against *base* unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else base / p


# ============================================================================
# ACE MODEL MATH
# ============================================================================

ACE_PARAM_NAMES = ["A", "C", "E"]

# The four summary statistics the model is identified from.
COV_FEATURE_NAMES = ["mz_var", "mz_cov", "dz_var", "dz_cov"]

# How the `*_var` features are reduced from a 2x2 sample covariance matrix.
# Stamped into each run's config.json so downstream scripts can detect runs
# trained before the sufficient reduction was adopted.  See summarize_cov().
VAR_REDUCTION = "mean_diagonal"


def theoretical_covariances(A, C, E):
    """
    Expected (noise-free) MZ and DZ covariance matrices under the ACE model.

    MZ twins share 100% of additive genetic variance and 100% of the shared
    environment; DZ twins share 50% and 100% respectively:

        Var    = A + C + E        (both zygosities)
        Cov_MZ = A + C
        Cov_DZ = 0.5*A + C

    Returns
    -------
    (mz_cov, dz_cov) : two 2x2 numpy arrays
    """
    V = A + C + E
    mz = np.array([[V, A + C],
                   [A + C, V]])
    dz = np.array([[V, 0.5 * A + C],
                   [0.5 * A + C, V]])
    return mz, dz


def summarize_cov(S):
    """
    Reduce a 2x2 twin sample covariance matrix to its sufficient statistic
    ``(var, cov)``.

    Under the exchangeable twin model the population matrix is compound
    symmetric, Sigma = [[v, c], [c, v]], and

        tr(Sigma^-1 S) = ( v*(S11 + S22) - 2*c*S12 ) / (v^2 - c^2)

    so the Gaussian likelihood depends on S only through ``S11 + S22`` and
    ``S12``.  The sufficient reduction is therefore the MEAN of the two
    diagonal entries — both estimate the same phenotypic variance v — paired
    with the off-diagonal.

    Using S11 alone (as this pipeline originally did) discards the second,
    partially independent estimate of v.  The variance of the resulting
    estimator is inflated by a factor 2/(1 + rho^2), where rho = c/v is the
    twin correlation: a full factor 2 (i.e. sqrt(2) in SD) when rho = 0,
    shrinking to no loss at all as rho approaches 1.

    Args:
        S: 2x2 sample covariance matrix.
    Returns:
        (var, cov) as floats.
    """
    S = np.asarray(S)
    return float(0.5 * (S[0, 0] + S[1, 1])), float(S[0, 1])


def simulate_covariances(A, C, E, N_pairs):
    """
    Draw ``N_pairs`` MZ and DZ twin pairs from the theoretical ACE covariance
    matrices and return their *sample* covariance matrices.

    This is the simulator at the heart of the NPE pipeline: it maps a
    parameter vector theta = (A, C, E) plus a sample size to a noisy
    observation x, exactly as a real study would.

    Returns
    -------
    (mz_cov_sim, dz_cov_sim) : two 2x2 numpy arrays
    """
    mz_cov, dz_cov = theoretical_covariances(A, C, E)

    df_mz = np.random.multivariate_normal(mean=[0, 0], cov=mz_cov, size=N_pairs)
    df_dz = np.random.multivariate_normal(mean=[0, 0], cov=dz_cov, size=N_pairs)

    return np.cov(df_mz, rowvar=False), np.cov(df_dz, rowvar=False)


def generate_training_data(n_samples=20000, n_pairs_options=None, seed=42):
    """
    Generate a (theta, x) training set for the NPE.

    For each sample:
      1. Draw A, C, E independently and uniformly from [0, 1].  No sum-to-1
         constraint is imposed, so the network learns the unstandardized
         scale as well as the ratios.
      2. Pick N_pairs — fixed if ``n_pairs_options`` holds a single value,
         otherwise drawn at random from it.
      3. Simulate twin data and reduce each 2x2 sample covariance matrix to
         its sufficient statistic via ``summarize_cov`` (mean of the two
         diagonal entries, plus the off-diagonal).

    Args:
        n_samples:       Number of training samples to generate.
        n_pairs_options: A single int (fixed N for all samples) or a list of
                         ints to draw from randomly.  Defaults to
                         [50, 100, 200, 500, 1000, 2000, 5000, 20000] — this
                         range covers every sample size STEP 05/06/07
                         evaluate at, so a with-N model trained on the
                         default never has to extrapolate its se_proxy
                         (1/sqrt(N)) feature beyond what it saw in training.
        seed:            NumPy random seed.

    Returns:
        pd.DataFrame with columns
        [mz_var, mz_cov, dz_var, dz_cov, N_pairs, log_N_pairs, se_proxy, A, C, E]
    """
    if n_pairs_options is None:
        n_pairs_options = [50, 100, 200, 500, 1000, 2000, 5000, 20000]

    if isinstance(n_pairs_options, (int, np.integer)):
        n_pairs_options = [int(n_pairs_options)]
    fixed_n = len(n_pairs_options) == 1

    np.random.seed(seed)

    records = []
    for i in range(n_samples):
        A = float(np.random.uniform(0, 1))
        C = float(np.random.uniform(0, 1))
        E = float(np.random.uniform(0, 1))

        N_pairs = (n_pairs_options[0] if fixed_n
                   else int(np.random.choice(n_pairs_options)))

        S_mz, S_dz = simulate_covariances(A, C, E, N_pairs)
        mz_var, mz_cov = summarize_cov(S_mz)
        dz_var, dz_cov = summarize_cov(S_dz)

        records.append({
            "mz_var":      mz_var,
            "mz_cov":      mz_cov,
            "dz_var":      dz_var,
            "dz_cov":      dz_cov,
            "N_pairs":     N_pairs,
            "log_N_pairs": np.log(N_pairs),
            "se_proxy":    1.0 / np.sqrt(N_pairs),
            "A": A, "C": C, "E": E,
        })

        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples...")

    return pd.DataFrame(records)


# ============================================================================
# EMBEDDING NETWORK
# ============================================================================

class ACEEmbeddingNet(nn.Module):
    """
    Lightweight embedding network for NPE on the ACE model.

    Currently BYPASSED — ``02_train_npe.py`` conditions the flow on the raw
    standardized features via ``nn.Identity()``.  See README.md,
    "Why no embedding network?".

    Uses LayerNorm rather than BatchNorm1d so that it behaves identically
    with batch_size=1, which is how ``posterior.sample()`` calls it.
    The ``output_dim`` attribute is required by sbi.
    """

    def __init__(self, n_features=4, hidden_sizes=None, dropout_rate=0.2):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64, 32]

        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_size, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
            in_size = h

        self.network = nn.Sequential(*layers)
        self.output_dim = in_size   # required by sbi

    def forward(self, x):
        return self.network(x)


# ============================================================================
# POSTERIOR HELPERS
# ============================================================================

def map_from_samples(samples: np.ndarray) -> np.ndarray:
    """
    Approximate the MAP (mode) for each parameter independently using a 1-D
    kernel density estimate over the posterior draws.

    Args:
        samples: (n_samples, n_params) array of posterior draws.
    Returns:
        (n_params,) array of MAP estimates.
    """
    map_est = np.empty(samples.shape[1])
    for i in range(samples.shape[1]):
        kde = gaussian_kde(samples[:, i])
        xs = np.linspace(samples[:, i].min(), samples[:, i].max(), 1000)
        map_est[i] = xs[np.argmax(kde(xs))]
    return map_est


def n_pairs_encoding(feature_cols):
    """
    Work out how (and whether) N_pairs entered a trained model's features.

    Returns
    -------
    (include_n_pairs, use_log_n_pairs, use_se_proxy) : three bools
    """
    include = any(c in feature_cols for c in ("N_pairs", "log_N_pairs", "se_proxy"))
    use_se_proxy = "se_proxy" in feature_cols
    use_log = (not use_se_proxy) and ("log_N_pairs" in feature_cols)
    return include, use_log, use_se_proxy


def describe_n_encoding(feature_cols) -> str:
    """Human-readable description of a run's N_pairs encoding."""
    include, use_log, use_se_proxy = n_pairs_encoding(feature_cols)
    if use_se_proxy:
        return "se_proxy (1/sqrt(N))"
    if use_log:
        return "log(N)"
    return "raw N" if include else "none"


def build_features(mz_var, mz_cov, dz_var, dz_cov, N, feature_cols):
    """
    Assemble a feature vector in the order a trained run expects.

    The four covariance statistics always come first; an N_pairs-derived
    feature is appended only if the run was trained with one, encoded the
    same way it was during training.

    Returns
    -------
    (1, n_features) float32 array, ready for the run's feature scaler.
    """
    include, use_log, use_se_proxy = n_pairs_encoding(feature_cols)

    feats = [mz_var, mz_cov, dz_var, dz_cov]
    if include:
        if use_se_proxy:
            feats.append(1.0 / np.sqrt(float(N)))
        elif use_log:
            feats.append(np.log(float(N)))
        else:
            feats.append(float(N))

    return np.array(feats, dtype=np.float32).reshape(1, -1)


def load_posterior(model_dir):
    """
    Load a trained NPE run: the posterior, its feature scaler, and its config.

    Args:
        model_dir: Path to a run directory produced by 02_train_npe.py
                   (relative paths resolve against results/models/).

    Returns
    -------
    dict with keys: posterior, scaler, config, feature_cols, param_names,
    model_dir.

    Raises
    ------
    FileNotFoundError if the run is missing config.json or posterior.pkl.
    """
    model_dir = resolve(model_dir, MODELS_DIR)

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(config_path) as f:
        config = json.load(f)

    feature_cols = config.get("feature_cols", list(COV_FEATURE_NAMES))
    param_names  = config.get("param_names", ACE_PARAM_NAMES)

    # Runs trained before the sufficient-variance reduction was adopted used
    # S[0,0] alone as the *_var features.  Feeding them today's mean-of-
    # diagonal features is a train/test mismatch: same expectation, but the
    # feature is less noisy than the run was trained to expect, so posteriors
    # come out miscalibrated (too wide).  Warn rather than fail — the run is
    # still loadable and its own metrics remain valid.
    run_reduction = config.get("var_feature")
    if run_reduction != VAR_REDUCTION:
        msg = (
            f"Run '{model_dir.name}' was trained with var_feature="
            f"{run_reduction!r}, but this code produces {VAR_REDUCTION!r} "
            "(mean of the two diagonal entries). Its features are NOT "
            "comparable with freshly simulated ones — retrain with "
            "02_train_npe.py before using it for inference."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        # The pipeline scripts call warnings.filterwarnings('ignore') to mute
        # sbi/torch chatter, which would swallow the warning above — precisely
        # where it matters most. Print unconditionally so it cannot be lost.
        print(f"\n  !! STALE RUN: {msg}\n", file=sys.stderr)

    posterior_path = model_dir / "posterior.pkl"
    if not posterior_path.exists():
        raise FileNotFoundError(
            f"posterior.pkl not found in {model_dir}. Run 02_train_npe.py first."
        )
    with open(posterior_path, "rb") as f:
        posterior = pickle.load(f)

    scaler = joblib.load(model_dir / "feature_scaler.pkl")

    if hasattr(posterior, "_neural_net"):
        posterior._neural_net.eval()

    return {
        "posterior":    posterior,
        "scaler":       scaler,
        "config":       config,
        "feature_cols": feature_cols,
        "param_names":  param_names,
        "model_dir":    model_dir,
    }


def posterior_stats(posterior, x_scaled_1d, n_samples, ci=(2.5, 97.5)):
    """
    Draw ``n_samples`` posterior draws for one scaled observation.

    Args:
        posterior:   trained sbi posterior object
        x_scaled_1d: 1-D array of scaled features for one observation
        n_samples:   number of posterior draws
        ci:          percentile pair for the credible interval (default 95%)

    Returns
    -------
    (mean, std, map, ci_lo, ci_hi) : five (n_params,) arrays

    The interval bounds are what STEP 07 needs to measure coverage — the
    fraction of replicates whose true value falls inside the reported CI.
    """
    x_t = torch.FloatTensor(np.asarray(x_scaled_1d)).unsqueeze(0)
    with torch.no_grad():
        samples = posterior.sample((n_samples,), x=x_t, show_progress_bars=False)
    s = samples.cpu().numpy()
    return (s.mean(axis=0), s.std(axis=0), map_from_samples(s),
            np.percentile(s, ci[0], axis=0), np.percentile(s, ci[1], axis=0))
