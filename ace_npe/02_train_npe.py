"""
STEP 02 — Neural Posterior Estimation (NPE) for ACE Parameter Estimation

Uses sbi (SNPE-C / APT) to learn the full posterior over the A, C, E variance
components under the Dirichlet prior used by STEP 01. The flow learns the two
additive-log-ratio coordinates log(A/E) and log(C/E); returned posterior draws
are transformed back to positive A, C, E values that sum to fixed V.

Architecture overview
---------------------
1. **Conditioning input**: the 4 (or 5) summary statistics, standardized by a
   ``StandardScaler`` fitted on the training split.  These are fed to the flow
   DIRECTLY via ``nn.Identity()`` — no embedding network is used.  See the
   "Why no embedding network?" section of README.md for the reasoning;
   ``ace_model.ACEEmbeddingNet`` remains available as a drop-in replacement.

2. **Normalizing Flow** (Neural Spline Flow, via ``sbi``):
   Conditions on that vector and outputs the full posterior over the 3 ACE
   parameters.

Input features  (4):  mz_var, mz_cov, dz_var, dz_cov
                      Optionally add an N_pairs encoding with --include_n_pairs
Learned targets (2):  log(A/E), log(C/E)
Posterior output (3): A, C, E, constrained so A+C+E=V

Usage:
    # 1. Generate training data first
    python 01_generate_training_data.py --n_samples 20000

    # 2. Train (paths are relative to data/ and results/models/)
    python 02_train_npe.py --data ace_training_data_N2000.csv --epochs 200 \
                           --device cpu --output no_n_pairs_wideprior

    # 3. Optionally add an N_pairs-derived feature
    python 02_train_npe.py --data ace_training_data.csv --include_n_pairs \
                           --epochs 500 --device cpu --output se_proxy

    # 4. Use the same non-default Dirichlet prior as STEP 01, if applicable
    python 02_train_npe.py --data ace_training_data_N20000.csv --epochs 500 \
                           --device cpu --dirichlet_alpha 2 2 2 \
                           --output no_n_pairs_dirichlet
"""

import sys
import math
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_model import (
    ACE_PARAM_NAMES,
    COV_FEATURE_NAMES,
    ACEEmbeddingNet,  # noqa: F401 — used by the commented-out embedding path below
    DirichletACEPosterior,
    DirichletALRPrior,
    DATA_DIR,
    MODELS_DIR,
    VAR_REDUCTION,
    ace_to_alr,
    map_from_samples,
    resolve,
)

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_clean_data(data_path):
    print("\n" + "="*70)
    print("LOADING AND CLEANING DATA")
    print("="*70)

    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} columns")

    initial_samples = len(df)
    feature_cols  = list(COV_FEATURE_NAMES)
    required_cols = feature_cols + ACE_PARAM_NAMES

    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Required columns missing from data: {missing_required}")

    df = df.dropna(subset=required_cols).copy()
    print(f"✓ Removed {initial_samples - len(df)} rows with missing values")

    invalid_mz = (df['mz_cov'].abs() > df['mz_var'].abs())
    invalid_dz = (df['dz_cov'].abs() > df['dz_var'].abs())
    n_invalid = (invalid_mz | invalid_dz).sum()
    if n_invalid > 0:
        df = df[~(invalid_mz | invalid_dz)].copy()
        print(f"✓ Removed {n_invalid} rows with invalid covariance matrices")

    print(f"\nFinal sample size: {len(df)}")
    print("="*70)
    return df


# ============================================================================
# NPE EVALUATION
# ============================================================================

def evaluate_npe(posterior, X_test, y_test, param_names, output_dir,
                 n_posterior_samples=500, device=None, n_eval=500):
    """
    Evaluate the trained NPE on the held-out test set.

    For each test observation, draw ``n_posterior_samples`` samples from the
    posterior and use their mean as the point estimate and std as uncertainty.

    Evaluation always runs on CPU regardless of training device: MPS has high
    per-kernel-launch overhead that makes serial single-observation sampling
    much slower than CPU.
    """
    print("\n" + "="*70)
    print(f"EVALUATING NPE ON TEST SET  (n_posterior_samples={n_posterior_samples})")
    print("="*70)

    # Always evaluate on CPU — serial single-obs sampling is faster there
    eval_posterior = posterior
    try:
        eval_posterior = posterior.set_default_x(None)  # reset any cached x
    except Exception:
        pass

    n_test = min(len(X_test), n_eval)
    if n_test < len(X_test):
        print(f"  ⚠ Evaluating on first {n_test}/{len(X_test)} test samples for speed.")
    print(f"  Running on: cpu  (faster than MPS/CUDA for serial single-obs sampling)")

    if hasattr(eval_posterior, '_neural_net'):
        eval_posterior._neural_net.eval()

    pred_means, pred_stds, pred_maps = [], [], []
    for i in range(n_test):
        x_obs = torch.FloatTensor(X_test[i]).unsqueeze(0)  # keep on CPU
        with torch.no_grad():
            samples = eval_posterior.sample(
                (n_posterior_samples,), x=x_obs, show_progress_bars=False
            )
        s = samples.cpu().numpy()
        pred_means.append(s.mean(0))
        pred_stds.append(s.std(0))
        pred_maps.append(map_from_samples(s))

    pred_means = np.array(pred_means)
    pred_stds  = np.array(pred_stds)
    pred_maps  = np.array(pred_maps)
    y_true     = y_test[:n_test]

    results = {}
    print(f"\n{'Parameter':<22} {'R² (mean)':<12} {'R² (MAP)':<12} {'RMSE':<10} {'MAE':<10} {'Mean Posterior σ'}")
    print("-"*80)

    for i, param in enumerate(param_names):
        y_t  = y_true[:, i]
        y_p  = pred_means[:, i]
        y_m  = pred_maps[:, i]
        y_s  = pred_stds[:, i]
        r2      = r2_score(y_t, y_p)
        r2_map  = r2_score(y_t, y_m)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae  = mean_absolute_error(y_t, y_p)
        results[param] = dict(r2=r2, r2_map=r2_map, rmse=rmse, mae=mae,
                              mean_sigma=float(y_s.mean()),
                              true=y_t, pred_mean=y_p, pred_std=y_s, pred_map=y_m)
        print(f"{param:<22} {r2:<12.4f} {r2_map:<12.4f} {rmse:<10.4f} {mae:<10.4f} {y_s.mean():.4f}")

    overall_r2   = r2_score(y_true.flatten(), pred_means.flatten())
    overall_rmse = np.sqrt(mean_squared_error(y_true.flatten(), pred_means.flatten()))
    overall_r2_map = r2_score(y_true.flatten(), pred_maps.flatten())
    print("-"*80)
    print(f"{'OVERALL (mean)':<22} {overall_r2:<12.4f} {overall_r2_map:<12.4f} {overall_rmse:<10.4f}")
    print("="*80 + "\n")

    metrics_dict = {
        p: {k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in vals.items()}
        for p, vals in results.items()
    }
    metrics_dict['overall'] = {'r2': float(overall_r2), 'r2_map': float(overall_r2_map),
                                'rmse': float(overall_rmse)}
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_npe_results(results, output_dir, param_names):
    """Prediction scatter plots (MAP vs True) and posterior uncertainty bar plot."""
    n_params = len(param_names)
    n_cols   = min(n_params, 4)
    n_rows   = math.ceil(n_params / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    for i, param in enumerate(param_names):
        ax    = axes[i]
        y_t   = results[param]['true']
        y_map = results[param]['pred_map']
        y_s   = results[param]['pred_std']
        r2    = results[param]['r2_map']

        ax.scatter(y_t, y_map, alpha=0.4, s=15, color='steelblue')
        lo, hi = min(y_t.min(), y_map.min()), max(y_t.max(), y_map.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Identity')
        ax.set_xlabel('True', fontsize=10)
        ax.set_ylabel('MAP Estimate', fontsize=10)
        ax.set_title(f'{param}  (R²={r2:.3f})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n_params, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.suptitle('NPE: MAP Estimate vs True', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_vs_true.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Posterior uncertainty bar chart
    fig, ax = plt.subplots(figsize=(max(6, n_params * 2), 4))
    mean_sigmas = [results[p]['mean_sigma'] for p in param_names]
    bars = ax.bar(param_names, mean_sigmas, color='steelblue', edgecolor='white')
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Mean Posterior σ')
    ax.set_title('Posterior Uncertainty per Parameter')
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, mean_sigmas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / 'posterior_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved prediction plots to {output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train NPE (sbi SNPE-C) for ACE parameter estimation'
    )
    parser.add_argument('--data', type=str, default='ace_training_data.csv',
                        help='Training CSV, relative to data/ '
                             '(default: ace_training_data.csv)')
    parser.add_argument('--output', type=str, default='default',
                        help='Run directory, relative to results/models/ '
                             '(default: default)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Maximum training epochs (default: 500)')
    parser.add_argument('--stop_after_epochs', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Training mini-batch size (default: 1024)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='AdamW weight decay (default: 1e-4)')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[64, 64, 32],
                        help='Embedding network hidden layer widths (default: 64 64 32)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--flow_type', type=str, default='nsf',
                        choices=['nsf', 'maf', 'maf_rqs', 'mdn'],
                        help='Normalizing flow type (default: nsf)')
    parser.add_argument('--flow_hidden', type=int, default=64,
                        help='Hidden features within the flow (default: 64)')
    parser.add_argument('--flow_transforms', type=int, default=5,
                        help='Number of flow transforms (default: 5)')
    parser.add_argument('--total_variance', type=float, default=1.0,
                        help='Fixed V=A+C+E used in STEP 01 (default: 1)')
    parser.add_argument('--dirichlet_alpha', type=float, nargs=3,
                        default=[1.0, 1.0, 1.0],
                        metavar=('ALPHA_A', 'ALPHA_C', 'ALPHA_E'),
                        help='Dirichlet concentration parameters used in STEP 01 '
                             '(default: 1 1 1)')
    parser.add_argument('--n_posterior_samples', type=int, default=500,
                        help='Posterior samples per test observation for evaluation (default: 500)')
    parser.add_argument('--n_eval', type=int, default=200,
                        help='Number of test observations to evaluate (default: 200)')
    parser.add_argument('--include_n_pairs', action='store_true',
                        help='Include N_pairs as an input feature')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, mps (default: auto)')
    args = parser.parse_args()

    if args.total_variance <= 0:
        parser.error('--total_variance must be positive')
    if any(alpha <= 0 for alpha in args.dirichlet_alpha):
        parser.error('--dirichlet_alpha values must be positive')

    data_path  = resolve(args.data, DATA_DIR)
    output_dir = resolve(args.output, MODELS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("STEP 02 — ACE NEURAL POSTERIOR ESTIMATION (SNPE-C)")
    print("="*70)
    print(f"Output: {output_dir}")
    print("="*70)

    # ---- Device ----
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    # ---- Load data ----
    df = load_and_clean_data(data_path)

    feature_cols = list(COV_FEATURE_NAMES)
    if args.include_n_pairs:
        if 'se_proxy' in df.columns:
            feature_cols.append('se_proxy')
            print("  Including se_proxy (1/√N_pairs) as feature")
        elif 'log_N_pairs' in df.columns:
            df['se_proxy'] = 1.0 / np.sqrt(np.exp(df['log_N_pairs']))
            feature_cols.append('se_proxy')
            print("  Including se_proxy (1/√N_pairs) as feature  [derived from log_N_pairs]")
        elif 'N_pairs' in df.columns:
            df['se_proxy'] = 1.0 / np.sqrt(df['N_pairs'])
            feature_cols.append('se_proxy')
            print("  Including se_proxy (1/√N_pairs) as feature  [derived from N_pairs]")

    X = df[feature_cols].values
    y = df[ACE_PARAM_NAMES].values.astype(np.float32)

    if 'simulation_scheme' in df.columns and not (
        df['simulation_scheme'] == 'dirichlet'
    ).all():
        raise ValueError(
            "STEP 02 now accepts Dirichlet training data only; "
            "simulation_scheme contains another value"
        )
    if np.any(y <= 0):
        raise ValueError('Dirichlet training targets A, C, E must all be positive')
    observed_v = y.sum(axis=1)
    max_sum_error = float(np.max(np.abs(observed_v - args.total_variance)))
    if max_sum_error > 1e-5:
        raise ValueError(
            f'Training targets do not satisfy A+C+E={args.total_variance:g}; '
            f'maximum absolute error is {max_sum_error:.3e}. '
            'Use matching --total_variance values in STEP 01 and STEP 02.'
        )
    theta = ace_to_alr(y)
    print(
        f"  Dirichlet target check: max |A+C+E-V|={max_sum_error:.3e}; "
        "learning log(A/E), log(C/E)"
    )

    # ---- Train/val/test split (70/15/15) ----
    n_samples = len(df)
    rng       = np.random.default_rng(42)
    indices   = rng.permutation(n_samples)
    n_train   = int(n_samples * 0.70)
    n_val     = int(n_samples * 0.15)

    X_train = X[indices[:n_train]]
    theta_train = theta[indices[:n_train]]
    X_val   = X[indices[n_train:n_train + n_val]]
    theta_val = theta[indices[n_train:n_train + n_val]]
    X_test  = X[indices[n_train + n_val:]]
    y_test  = y[indices[n_train + n_val:]]
    print(f"\nData split: {len(X_train)} train | {len(X_val)} val | {len(X_test)} test")

    # ---- Scale features (ALR theta is not scaled here; sbi handles it) ----
    feature_scaler = StandardScaler()
    X_train_s = feature_scaler.fit_transform(X_train)
    X_val_s   = feature_scaler.transform(X_val)
    X_test_s  = feature_scaler.transform(X_test)
    joblib.dump(feature_scaler, output_dir / 'feature_scaler.pkl')

    # ---- Define the exact STEP 01 Dirichlet prior in ALR coordinates ----
    print("\n" + "="*70)
    print("PRIOR DEFINITION (EXACT DIRICHLET IN ALR COORDINATES)")
    print("="*70)
    print(f"  Dirichlet alpha : {tuple(args.dirichlet_alpha)}")
    print(f"  Fixed V         : {args.total_variance:g}")
    print("  Flow targets    : log(A/E), log(C/E)")
    print("  Returned draws  : positive A, C, E with A+C+E=V")
    prior = DirichletALRPrior(args.dirichlet_alpha, device=str(device))

    # ---- Build density estimator ----
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)

    n_features = X_train.shape[1]

    # NOTE: the flow conditions on the standardized summary statistics
    # DIRECTLY (nn.Identity), not on a learned embedding.  With only 4-5
    # near-sufficient statistics there is nothing to compress, and an
    # unnecessary bottleneck can only discard information.  See README.md,
    # "Why no embedding network?".  To re-enable the embedding net, swap the
    # two lines below — note this invalidates every previously trained run.
    #
    #   embedding_net = ACEEmbeddingNet(n_features=n_features,
    #                                   hidden_sizes=args.hidden_sizes,
    #                                   dropout_rate=args.dropout)
    embedding_net = nn.Identity()

    print(f"\nConditioning input (no embedding network):")
    print(f"  Input features : {n_features}  → fed directly to the flow")
    print(f"  Standardizer   : StandardScaler (fit on the training split)")
    print(f"  Embedding      : nn.Identity()   [ACEEmbeddingNet available but bypassed]")

    density_estimator_fn = posterior_nn(
        model=args.flow_type,
        embedding_net=embedding_net,
        hidden_features=args.flow_hidden,
        num_transforms=args.flow_transforms,
        z_score_theta='independent',
        z_score_x='independent',
    )

    print(f"\nNormalizing Flow:")
    print(f"  Type           : {args.flow_type.upper()}")
    print(f"  Hidden features: {args.flow_hidden}")
    print(f"  Transforms     : {args.flow_transforms}")

    # ---- Train ----
    print("\n" + "="*70)
    print("TRAINING (SNPE-C)")
    print("="*70)

    inference = SNPE(
        prior=prior,
        density_estimator=density_estimator_fn,
        device=str(device),
    )

    X_all = np.vstack([X_train_s, X_val_s])
    theta_all = np.vstack([theta_train, theta_val])

    inference.append_simulations(
        theta=torch.tensor(theta_all, dtype=torch.float32),
        x=torch.tensor(X_all, dtype=torch.float32),
    )

    density_estimator = inference.train(
        training_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_num_epochs=args.epochs,
        stop_after_epochs=args.stop_after_epochs,
        show_train_summary=True,
        validation_fraction=len(X_val_s) / len(X_all),
    )

    # ---- Build and save posterior ----
    latent_posterior = inference.build_posterior(density_estimator)
    latent_posterior.to('cpu')
    posterior = DirichletACEPosterior(
        latent_posterior, total_variance=args.total_variance
    )
    print("\n✓ Posterior built successfully (ALR draws are returned as A, C, E).")

    torch.save(
        {
            'density_estimator_state_dict': density_estimator.state_dict(),
            'prior_type': 'exact_dirichlet_ALR',
            'dirichlet_alpha': list(args.dirichlet_alpha),
            'total_variance': args.total_variance,
            'theta_parameterization': 'ALR: log(A/E), log(C/E); E reconstructed',
        },
        output_dir / 'density_estimator.pt',
    )
    with open(output_dir / 'posterior.pkl', 'wb') as f:
        pickle.dump(posterior, f)
    print(f"✓ Saved posterior  → {output_dir / 'posterior.pkl'}")

    # ---- Evaluate and plot ----
    results = evaluate_npe(
        posterior, X_test_s, y_test, ACE_PARAM_NAMES,
        output_dir, n_posterior_samples=args.n_posterior_samples,
        device=device, n_eval=args.n_eval,
    )
    plot_npe_results(results, output_dir, ACE_PARAM_NAMES)

    # ---- Save config ----
    config = {
        'model_type': 'NPE',
        'flow_type': args.flow_type,
        'flow_hidden': args.flow_hidden,
        'flow_transforms': args.flow_transforms,
        'n_features': n_features,
        'feature_cols': feature_cols,
        # How *_var was reduced from the 2x2 sample covariance matrix.
        # Downstream scripts refuse to trust a run whose value differs.
        'var_feature': VAR_REDUCTION,
        # NOTE: hidden_sizes/dropout_rate configure ACEEmbeddingNet, which is
        # currently bypassed (see MODEL ARCHITECTURE above). Recorded for
        # provenance only — they do not affect the trained model.
        'hidden_sizes': args.hidden_sizes,
        'dropout_rate': args.dropout,
        'param_names': ACE_PARAM_NAMES,
        'simulation_scheme': 'dirichlet',
        'prior_type': 'exact_dirichlet_ALR',
        'dirichlet_alpha': list(args.dirichlet_alpha),
        'total_variance': args.total_variance,
        'theta_parameterization': 'ALR: log(A/E), log(C/E); E reconstructed',
        'latent_param_names': ['log_A_over_E', 'log_C_over_E'],
        'training_rows': len(X_train),
        'validation_rows': len(X_val),
        'test_rows': len(X_test),
        'source_data': str(data_path),
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved config     → {output_dir / 'config.json'}")

    print("\n✓ NPE Training complete!\n")


if __name__ == "__main__":
    main()
