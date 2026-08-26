"""
STEP 01 — Generate ACE Training Data

Simulates MZ and DZ twin covariance matrices for random (A, C, E) draws and
writes them to a CSV that STEP 02 trains on.  The heavy lifting lives in
``ace_model.generate_training_data``; this file is the command-line front end.

The ACE model:
  - A: additive genetic variance (heritability)
  - C: shared/common environment variance
  - E: unique environment variance (+ measurement error)

Theoretical covariance matrices:
  MZ twins: Var = A+C+E, Cov = A+C      (share 100% genes + 100% shared env)
  DZ twins: Var = A+C+E, Cov = 0.5A+C   (share  50% genes + 100% shared env)

Features saved:
  mz_var, mz_cov, dz_var, dz_cov  (unique elements of the 2x2 cov matrices)
  N_pairs, log_N_pairs, se_proxy  (three encodings of the sample size)

Targets saved:
  A, C, E

Usage:
    # Default: draw N randomly from [50, 100, 200, 500, 1000, 2000, 5000]
    python 01_generate_training_data.py --n_samples 20000

    # Fixed N for all samples (useful for training a no-N model)
    python 01_generate_training_data.py --n_pairs 2000 --n_samples 20000 \
                                        --output ace_training_data_N2000.csv

    # Draw randomly from a custom vector
    python 01_generate_training_data.py --n_pairs 200 500 1000 2000
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_model import DATA_DIR, generate_training_data, resolve


def main():
    parser = argparse.ArgumentParser(description='Generate ACE training data')
    parser.add_argument('--n_samples', type=int, default=20000,
                        help='Number of training samples (default: 20000)')
    parser.add_argument('--n_pairs', type=int, nargs='+', default=None,
                        help='Twin-pair sample size(s). Pass a single integer '
                             'to fix N for all samples (e.g. --n_pairs 500), '
                             'or multiple integers to draw randomly from that '
                             'vector (e.g. --n_pairs 50 100 200 500 1000 2000). '
                             'Defaults to [50, 100, 200, 500, 1000, 2000, 5000].')
    parser.add_argument('--output', type=str, default='ace_training_data.csv',
                        help='Output CSV filename, relative to data/ '
                             '(default: ace_training_data.csv)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = resolve(args.output, DATA_DIR)

    print("=" * 60)
    print("STEP 01 — GENERATING ACE TRAINING DATA")
    print("=" * 60)
    print(f"  Samples:    {args.n_samples}")
    print(f"  N pairs:    {args.n_pairs if args.n_pairs is not None else '[50,100,200,500,1000,2000,5000] (default)'}")
    print(f"  Seed:       {args.seed}")
    print(f"  Output:     {output_path}")
    print()

    df = generate_training_data(n_samples=args.n_samples,
                                n_pairs_options=args.n_pairs,
                                seed=args.seed)

    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df)} rows to {output_path}")
    print(f"\nColumn summary:")
    print(df.describe().to_string())

    # Quick sanity check: theoretical MZ cov ≈ A+C, DZ cov ≈ 0.5A+C
    print("\nSanity check (mean absolute deviation from theoretical values):")
    print(f"  |mz_cov - (A+C)|   mean: {(df['mz_cov'] - (df['A'] + df['C'])).abs().mean():.4f}")
    print(f"  |dz_cov - (.5A+C)| mean: {(df['dz_cov'] - (0.5*df['A'] + df['C'])).abs().mean():.4f}")


if __name__ == "__main__":
    main()
