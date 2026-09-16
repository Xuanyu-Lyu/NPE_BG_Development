"""STEP 01b -- simulate ACE training data under three parameter distributions.

For every scheme, a separate dataset is written for each fixed N.  The default
therefore creates 3 x 6 files for N in [50, 100, 500, 1000, 5000, 20000].
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_model import DATA_DIR, resolve
from ace_prior_comparison import (
    DEFAULT_N_PAIRS,
    SCHEMES,
    SCHEME_LABELS,
    generate_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate matched ACE training datasets for three simulation schemes"
    )
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument(
        "--n_pairs",
        type=int,
        nargs="+",
        default=None,
        help="Fixed-N datasets to create (default: 50 100 500 1000 5000 20000)",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=SCHEMES,
        default=list(SCHEMES),
        help="Simulation arms to generate",
    )
    parser.add_argument(
        "--total_variance",
        type=float,
        default=1.0,
        help="A+C+E for the two fixed-sum arms (default: 1)",
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("ALPHA_A", "ALPHA_C", "ALPHA_E"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="prior_comparison",
        help="Directory relative to ace_npe/data/ (default: prior_comparison)",
    )
    parser.add_argument(
        "--skip_manifest",
        action="store_true",
        help=(
            "Write only the requested CSV files and leave an existing manifest "
            "unchanged. Useful when adding one fixed-N dataset."
        ),
    )
    args = parser.parse_args()

    if args.n_samples <= 0:
        parser.error("--n_samples must be positive")
    if args.total_variance <= 0:
        parser.error("--total_variance must be positive")
    if any(alpha <= 0 for alpha in args.dirichlet_alpha):
        parser.error("--dirichlet_alpha values must be positive")

    n_options = tuple(args.n_pairs) if args.n_pairs is not None else DEFAULT_N_PAIRS
    if any(n < 2 for n in n_options):
        parser.error("--n_pairs values must be at least 2")

    output_dir = resolve(args.output_dir, DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("STEP 01b -- ACE PRIOR/SIMULATION COMPARISON DATA")
    print("=" * 72)
    print(f"Samples per scheme: {args.n_samples}")
    print(f"Fixed-N datasets:   {list(n_options)}")
    print(f"Fixed total V:      {args.total_variance}")
    print(f"Dirichlet alpha:    {args.dirichlet_alpha}")
    print(f"Output:             {output_dir}")

    files = {scheme: {} for scheme in args.schemes}
    for scheme in args.schemes:
        for n_pairs in n_options:
            print(f"\nGenerating {SCHEME_LABELS[scheme]}, fixed N={n_pairs} ...")
            df = generate_dataset(
                n_samples=args.n_samples,
                scheme=scheme,
                n_pairs_options=[n_pairs],
                seed=args.seed,
                total_variance=args.total_variance,
                dirichlet_alpha=args.dirichlet_alpha,
            )
            output_path = output_dir / f"ace_{scheme}_N{n_pairs}.csv"
            df.to_csv(output_path, index=False)
            files[scheme][str(n_pairs)] = output_path.name
            print(f"Saved {len(df)} rows -> {output_path}")
            print(df.describe().to_string())

    manifest = {
        "experiment": "ace_prior_comparison",
        "n_samples_per_scheme": args.n_samples,
        "fixed_n_pairs": list(n_options),
        "one_model_per_n": True,
        "n_is_model_feature": False,
        "seed": args.seed,
        "total_variance_fixed_sum": args.total_variance,
        "dirichlet_alpha": args.dirichlet_alpha,
        "schemes": {
            "independent_uniform": "A,C,E iid Uniform(0,1); no sum constraint",
            "normalized_uniform": (
                "U_A,U_C,U_E iid Uniform(0,1), then ACE=V*U/sum(U); "
                "deliberately not Dirichlet-uniform"
            ),
            "dirichlet": "ACE=V*Dirichlet(alpha_A,alpha_C,alpha_E)",
        },
        "files": files,
    }
    if args.skip_manifest:
        print("\nSkipped manifest update (--skip_manifest)")
    else:
        manifest_path = output_dir / "simulation_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nSaved manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
