"""
Generate BiSEMPGS Training Data

Simulates the 14×14 theoretical covariance matrix for random combinations of
bivariate SEM-PGS model parameters, adds realistic sampling noise by drawing
observations from a multivariate normal distribution, and saves the unique
upper-triangle covariance elements as features (inputs) and the true model
parameters as targets (outputs).

The 14 variables in the covariance matrix correspond to:
  Yp1, Yp2   – parent phenotypes (traits 1 & 2)
  Ym1, Ym2   – mate phenotypes
  Yo1, Yo2   – offspring phenotypes
  Tp1, Tp2   – paternal transmitted PGS
  NTp1, NTp2 – paternal non-transmitted PGS
  Tm1, Tm2   – maternal transmitted PGS
  NTm1, NTm2 – maternal non-transmitted PGS

Features saved (45 + 1 columns):
  <block>_ij  – the 45 unique elements drawn from the 12 distinct 2×2 blocks
               of the 14×14 sample covariance matrix (many blocks repeat
               across multiple positions and are recorded only once)
  N_obs       – number of simulated observations (controls estimation noise)

Targets saved (14 columns, prefixed with 'param_'):
  vg1, vg2, rg, re,
  prop_h2_latent1, prop_h2_latent2,
  am11, am12, am21, am22,
  f11, f12, f21, f22

Usage:
    python generate_training_data.py --n_samples 20000 \
                                     --output bisempgs_training_data.csv
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Allow importing bisempgs_model from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from bisempgs_model import (
    compute_cmatrix,
    simulate_sample_cov,
    unique_elements,
    unique_feature_names,
    N_UNIQUE_FEATURES,
)


# ---------------------------------------------------------------------------
# Parameter prior bounds  (uniform sampling)
# ---------------------------------------------------------------------------

PARAM_BOUNDS = {
    # Total SNP heritability for each trait
    "vg1":             (0.01, 0.99),
    "vg2":             (0.01, 0.99),
    # Genetic correlation between traits
    "rg":              (-0.99,  0.99),
    # Residual environmental correlation
    "re":              (-0.99,  0.99),
    # Fraction of heritability explained by the *latent* (unobserved) PGS
    "prop_h2_latent1": (0.01,  0.99),
    "prop_h2_latent2": (0.01,  0.99),
    # Assortative mating correlations (mate-correlation matrix)
    "am11":            (-0.70,  0.70),   # within-trait AM (trait 1)
    "am12":            (-0.70,  0.70),  # cross-trait AM  (Yp1 → Ym2)
    "am21":            (-0.70,  0.70),  # cross-trait AM  (Yp2 → Ym1)
    "am22":            (-0.70,  0.70),   # within-trait AM (trait 2)
    # Vertical transmission (VT) coefficients
    "f11":             (-0.25,  0.25),   # direct VT (trait 1)
    "f12":             (-0.25,  0.25),  # cross VT  (parent trait 2 → offspring trait 1)
    "f21":             (-0.25,  0.25),  # cross VT  (parent trait 1 → offspring trait 2)
    "f22":             (-0.25,  0.25),   # direct VT (trait 2)
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())   # 14 parameter names


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _sample_params(rng):
    """Draw one parameter vector uniformly from PARAM_BOUNDS."""
    return {name: float(rng.uniform(*bounds))
            for name, bounds in PARAM_BOUNDS.items()}


def generate_training_data(
    n_samples: int = 20_000,
    n_obs_options=None,
    seed: int = 42,
    gens: int = 15,
):
    """Generate training data for a BiSEMPGS neural network.

    For each accepted sample:
      1. Draw model parameters uniformly from PARAM_BOUNDS.
      2. Compute the 14×14 theoretical CMatrix (skips if invalid / non-PSD).
      3. Draw N_obs from *n_obs_options* and simulate a sample covariance.
      4. Record the 105 upper-triangle covariance elements + N_obs as
         features, and the 14 true parameters as targets.

    Parameters
    ----------
    n_samples : int
        Number of valid training samples to generate.
    n_obs_options : list[int] or None
        Sample sizes to draw from at random.  Larger values → less noise.
    seed : int
        Random seed for reproducibility.
    gens : int
        Number of AM/VT generations for the iterative model.

    Returns
    -------
    pd.DataFrame
        DataFrame with 106 feature columns (cov_* + N_obs) and
        14 target columns (param_*).
    """
    if n_obs_options is None:
        n_obs_options = [5_000, 10_000, 20_000, 50_000, 100_000, 200_000]

    rng        = np.random.default_rng(seed)
    feat_names = unique_feature_names()   # 45 column names

    records     = []
    n_attempts  = 0
    max_attempts = n_samples * 50      # up to 50 draws per valid sample

    print(f"Generating {n_samples:,} training samples …")

    while len(records) < n_samples:
        if n_attempts >= max_attempts:
            print(
                f"\n  WARNING: reached {max_attempts:,} attempts; "
                f"only {len(records):,} valid samples collected."
            )
            break

        n_attempts += 1
        params = _sample_params(rng)

        cmat = compute_cmatrix(**params, gens=gens)
        if cmat is None:
            continue   # invalid parameter combination — try again

        n_obs      = int(rng.choice(n_obs_options))
        sample_cov = simulate_sample_cov(cmat, n_obs)

        feats  = unique_elements(sample_cov)           # shape (45,)
        record = dict(zip(feat_names, feats))
        record["N_obs"] = n_obs
        for name in PARAM_NAMES:
            record[f"param_{name}"] = params[name]

        records.append(record)

        n_done = len(records)
        if n_done % 2_000 == 0:
            print(
                f"  {n_done:>6,} / {n_samples:,}  "
                f"(attempts: {n_attempts:,}, "
                f"accept rate: {n_done / n_attempts:.1%})"
            )

    accept_rate = len(records) / n_attempts if n_attempts > 0 else 0.
    print(
        f"\nDone — {len(records):,} samples in {n_attempts:,} attempts "
        f"(accept rate: {accept_rate:.1%})"
    )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate BiSEMPGS training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n_samples", type=int, default=20_000,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--output", type=str, default="bisempgs_training_data.csv",
        help="Output CSV filename",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--gens", type=int, default=20,
        help="Generations for the iterative AM/VT model",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path

    print("=" * 60)
    print("GENERATING BISEMPGS TRAINING DATA")
    print("=" * 60)
    print(f"  Samples : {args.n_samples:,}")
    print(f"  Seed    : {args.seed}")
    print(f"  Gens    : {args.gens}")
    print(f"  Output  : {output_path}")
    print()

    df = generate_training_data(
        n_samples=args.n_samples,
        seed=args.seed,
        gens=args.gens,
    )

    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df):,} rows  →  {output_path}")

    param_cols = [c for c in df.columns if c.startswith("param_")]
    feat_cols  = [c for c in df.columns if not c.startswith("param_") and c != "N_obs"]

    print(f"\nFeature summary  ({len(feat_cols)} unique-block columns, first 8 shown):")
    print(df[feat_cols[:8]].describe().round(4).to_string())

    print(f"\nParameter summary  ({len(param_cols)} params):")
    print(df[param_cols].describe().round(4).to_string())

    # Sanity check: diagonal of CMatrix should be ≈ VY elements (positive)
    diag_names = [f"cov_{v}_{v}" for v in
                  ["Yp1", "Yp2", "Ym1", "Ym2", "Yo1", "Yo2"]]
    print("\nSanity check — mean diagonal variances (should be > 0):")
    for col in diag_names:
        if col in df.columns:
            print(f"  mean({col}) = {df[col].mean():.4f}")


if __name__ == "__main__":
    main()
