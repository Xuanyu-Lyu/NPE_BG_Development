"""Shared utilities for the ACE prior/simulation comparison (scripts 01b--03b)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions import Distribution, constraints

from ace_model import summarize_cov, theoretical_covariances


SCHEMES = ("independent_uniform", "normalized_uniform", "dirichlet")
SCHEME_LABELS = {
    "independent_uniform": "Independent U(0,1)",
    "normalized_uniform": "Normalized U(0,1) (naive)",
    "dirichlet": "Dirichlet",
}
DEFAULT_N_PAIRS = (50, 100, 500, 1000, 5000, 20000)
FIXED_SUM_SCHEMES = frozenset({"normalized_uniform", "dirichlet"})


def validate_scheme(scheme: str) -> str:
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme {scheme!r}; expected one of {SCHEMES}")
    return scheme


def sample_ace(
    n_samples: int,
    scheme: str,
    rng: np.random.Generator,
    total_variance: float = 1.0,
    dirichlet_alpha=(1.0, 1.0, 1.0),
) -> np.ndarray:
    """Draw ACE values under one of the three comparison distributions.

    ``normalized_uniform`` deliberately implements the common but incorrect
    construction requested for this experiment: draw three iid U(0,1)
    variables and divide by their sum.  Its output is neither component-wise
    uniform nor Dirichlet-uniform on the simplex.
    """
    validate_scheme(scheme)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if total_variance <= 0:
        raise ValueError("total_variance must be positive")

    alpha = np.asarray(dirichlet_alpha, dtype=float)
    if alpha.shape != (3,) or np.any(alpha <= 0):
        raise ValueError("dirichlet_alpha must contain three positive values")

    if scheme == "independent_uniform":
        return rng.uniform(0.0, 1.0, size=(n_samples, 3))
    if scheme == "normalized_uniform":
        raw = rng.uniform(0.0, 1.0, size=(n_samples, 3))
        return total_variance * raw / raw.sum(axis=1, keepdims=True)
    return total_variance * rng.dirichlet(alpha, size=n_samples)


def _generate_independent_uniform_baseline(
    n_samples: int, n_pairs_options, seed: int
) -> pd.DataFrame:
    """Frozen copy of STEP 01's independent-uniform simulation procedure.

    This intentionally does not call ``generate_training_data``.  Keeping the
    baseline local prevents future changes to STEP 01 from silently changing
    this comparison, while preserving its current RNG order and calculations.
    """
    fixed_n = len(n_pairs_options) == 1
    np.random.seed(seed)

    records = []
    for i in range(n_samples):
        a = float(np.random.uniform(0, 1))
        c = float(np.random.uniform(0, 1))
        e = float(np.random.uniform(0, 1))
        n_pairs = (
            n_pairs_options[0]
            if fixed_n
            else int(np.random.choice(n_pairs_options))
        )

        mz_theory, dz_theory = theoretical_covariances(a, c, e)
        mz_pairs = np.random.multivariate_normal(
            mean=[0, 0], cov=mz_theory, size=n_pairs
        )
        dz_pairs = np.random.multivariate_normal(
            mean=[0, 0], cov=dz_theory, size=n_pairs
        )
        mz_var, mz_cov = summarize_cov(np.cov(mz_pairs, rowvar=False))
        dz_var, dz_cov = summarize_cov(np.cov(dz_pairs, rowvar=False))

        records.append(
            {
                "mz_var": mz_var,
                "mz_cov": mz_cov,
                "dz_var": dz_var,
                "dz_cov": dz_cov,
                "N_pairs": n_pairs,
                "log_N_pairs": np.log(n_pairs),
                "se_proxy": 1.0 / np.sqrt(n_pairs),
                "A": a,
                "C": c,
                "E": e,
                "V": a + c + e,
                "simulation_scheme": "independent_uniform",
            }
        )
        if (i + 1) % 5000 == 0:
            print(f"  independent_uniform: generated {i + 1}/{n_samples} samples")

    return pd.DataFrame.from_records(records)


def generate_dataset(
    n_samples: int,
    scheme: str,
    n_pairs_options=None,
    seed: int = 42,
    total_variance: float = 1.0,
    dirichlet_alpha=(1.0, 1.0, 1.0),
    n_pairs_values=None,
) -> pd.DataFrame:
    """Simulate one labeled ACE dataset using the standard covariance summary."""
    validate_scheme(scheme)
    if n_pairs_options is None:
        n_pairs_options = DEFAULT_N_PAIRS
    n_pairs_options = tuple(int(n) for n in n_pairs_options)
    if not n_pairs_options or any(n < 2 for n in n_pairs_options):
        raise ValueError("n_pairs_options must contain integers >= 2")

    # This is a frozen local implementation of the original STEP 01 procedure.
    # It does not inherit future behavior changes from generate_training_data.
    if scheme == "independent_uniform" and n_pairs_values is None:
        return _generate_independent_uniform_baseline(
            n_samples=n_samples, n_pairs_options=n_pairs_options, seed=seed
        )

    # Separate random streams prevent parameter sampling from changing the N
    # schedule or the covariance-noise stream.  This lets all three schemes use
    # exactly the same N values in 01b and 03b.
    theta_rng = np.random.default_rng(seed + 1)
    n_rng = np.random.default_rng(seed + 2)
    ace = sample_ace(
        n_samples,
        scheme,
        theta_rng,
        total_variance=total_variance,
        dirichlet_alpha=dirichlet_alpha,
    )

    if n_pairs_values is None:
        if len(n_pairs_options) == 1:
            n_values = np.full(n_samples, n_pairs_options[0], dtype=int)
        else:
            n_values = n_rng.choice(n_pairs_options, size=n_samples).astype(int)
    else:
        n_values = np.asarray(n_pairs_values, dtype=int)
        if n_values.shape != (n_samples,):
            raise ValueError("n_pairs_values must have shape (n_samples,)")
        if np.any(n_values < 2):
            raise ValueError("all n_pairs_values must be >= 2")

    # ace_model.simulate_covariances uses NumPy's legacy global RNG.  Seeding
    # it here keeps this helper compatible with the existing simulator while
    # retaining independent theta/N streams above.
    np.random.seed(seed + 3)
    records = []
    for i, ((a, c, e), n_pairs) in enumerate(zip(ace, n_values)):
        mz_theory, dz_theory = theoretical_covariances(a, c, e)
        mz_pairs = np.random.multivariate_normal([0.0, 0.0], mz_theory, int(n_pairs))
        dz_pairs = np.random.multivariate_normal([0.0, 0.0], dz_theory, int(n_pairs))
        mz_var, mz_cov = summarize_cov(np.cov(mz_pairs, rowvar=False))
        dz_var, dz_cov = summarize_cov(np.cov(dz_pairs, rowvar=False))
        records.append(
            {
                "mz_var": mz_var,
                "mz_cov": mz_cov,
                "dz_var": dz_var,
                "dz_cov": dz_cov,
                "N_pairs": int(n_pairs),
                "log_N_pairs": math.log(int(n_pairs)),
                "se_proxy": 1.0 / math.sqrt(int(n_pairs)),
                "A": float(a),
                "C": float(c),
                "E": float(e),
                "V": float(a + c + e),
                "simulation_scheme": scheme,
            }
        )
        if (i + 1) % 5000 == 0:
            print(f"  {scheme}: generated {i + 1}/{n_samples} samples")

    return pd.DataFrame.from_records(records)


def ace_to_latent(ace: np.ndarray, scheme: str) -> np.ndarray:
    """Map ACE targets to the coordinates learned by the flow."""
    validate_scheme(scheme)
    values = np.asarray(ace, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("ace must have shape (n, 3)")
    if scheme == "independent_uniform":
        return values.astype(np.float32)

    if np.any(values <= 0):
        raise ValueError("fixed-sum log-ratio targets must be strictly positive")
    proportions = values / values.sum(axis=1, keepdims=True)
    latent = np.log(proportions[:, :2] / proportions[:, 2:3])
    return latent.astype(np.float32)


def latent_to_ace(latent: torch.Tensor, scheme: str, total_variance=1.0) -> torch.Tensor:
    """Convert flow samples back to A, C, E."""
    validate_scheme(scheme)
    if scheme == "independent_uniform":
        return latent
    baseline = torch.zeros((*latent.shape[:-1], 1), dtype=latent.dtype, device=latent.device)
    proportions = torch.softmax(torch.cat([latent, baseline], dim=-1), dim=-1)
    return float(total_variance) * proportions


class SimplexALRPrior(Distribution):
    """Exact ALR-coordinate prior for either fixed-sum simulation scheme."""

    arg_constraints = {}
    support = constraints.independent(constraints.real, 1)
    has_rsample = False

    def __init__(
        self,
        scheme: str,
        dirichlet_alpha=(1.0, 1.0, 1.0),
        device="cpu",
        validate_args=None,
    ):
        if scheme not in FIXED_SUM_SCHEMES:
            raise ValueError(f"SimplexALRPrior requires a fixed-sum scheme, got {scheme!r}")
        self.scheme = scheme
        self.alpha = torch.as_tensor(dirichlet_alpha, dtype=torch.float32, device=device)
        if self.alpha.shape != (3,) or torch.any(self.alpha <= 0):
            raise ValueError("dirichlet_alpha must contain three positive values")
        super().__init__(torch.Size(), torch.Size([2]), validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape)
        if self.scheme == "dirichlet":
            composition = torch.distributions.Dirichlet(self.alpha).sample(shape)
        else:
            raw = torch.rand((*shape, 3), device=self.alpha.device, dtype=self.alpha.dtype)
            composition = raw / raw.sum(dim=-1, keepdim=True)
        return torch.log(composition[..., :2] / composition[..., 2:3])

    def log_prob(self, value):
        baseline = torch.zeros((*value.shape[:-1], 1), dtype=value.dtype, device=value.device)
        composition = torch.softmax(torch.cat([value, baseline], dim=-1), dim=-1)
        log_jacobian = torch.log(composition).sum(dim=-1)
        if self.scheme == "dirichlet":
            alpha = self.alpha.to(value.device, value.dtype)
            return torch.distributions.Dirichlet(alpha).log_prob(composition) + log_jacobian

        # If U_i iid U(0,1) and X_i=U_i/sum(U), then on the simplex
        # p_X(x)=1/(3*max(x)^3). Add the ALR inverse-Jacobian prod(x_i).
        log_simplex_density = -math.log(3.0) - 3.0 * torch.log(composition.max(dim=-1).values)
        return log_simplex_density + log_jacobian

    def to(self, device):
        self.alpha = self.alpha.to(device)
        return self


class ACEPosterior:
    """Expose latent fixed-sum posteriors through the usual three ACE outputs."""

    def __init__(self, latent_posterior, scheme: str, total_variance: float = 1.0):
        self.latent_posterior = latent_posterior
        self.scheme = validate_scheme(scheme)
        self.total_variance = float(total_variance)

    @property
    def _neural_net(self):
        return self.latent_posterior.posterior_estimator

    def sample(self, sample_shape=torch.Size(), x=None, **kwargs):
        latent = self.latent_posterior.sample(sample_shape, x=x, **kwargs)
        return latent_to_ace(latent, self.scheme, self.total_variance)

    def set_default_x(self, x):
        self.latent_posterior.set_default_x(x)
        return self

    def to(self, device):
        self.latent_posterior.to(device)
        return self


def model_dir_for(base_dir, scheme: str, n_pairs: int | None = None) -> Path:
    path = Path(base_dir) / validate_scheme(scheme)
    return path if n_pairs is None else path / f"N{int(n_pairs)}"
