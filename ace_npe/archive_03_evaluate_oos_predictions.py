"""
STEP 03 — Out-of-Sample Validation for the ACE NPE Model

Generates N fresh samples (default 200) that were NOT part of the training data,
draws posterior samples for each observation using the trained NPE posterior,
and evaluates bias by comparing posterior means vs true A, C, E values.

This is the standalone calibration check on a single trained run.  It reports
what STEP 02's held-out evaluation does not:
  - Posterior mean  (point estimate)
  - Posterior std   (uncertainty per observation)
  - 95% credible interval coverage
  - MAP estimates alongside posterior means
  - Bias and R² metrics

Outputs land in results/oos/<run-name>/ by default.

Usage:
    python 03_evaluate_oos_predictions.py
    python 03_evaluate_oos_predictions.py --n_samples 500 --model_dir se_proxy
"""

import sys
import math
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_model import (
    ACEEmbeddingNet,   # noqa: F401 — must be importable to unpickle some runs
    DEFAULT_MODEL_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    generate_training_data,
    load_posterior,
    map_from_samples,
    resolve,
)

warnings.filterwarnings('ignore')


# ============================================================================
# OUT-OF-SAMPLE VALIDATION
# ============================================================================

def run_oos_validation(n_samples=200, model_dir=DEFAULT_MODEL_DIR,
                       n_posterior_samples=500, seed=999, output_dir=None):
    """
    Generate n_samples fresh ACE samples (distinct seed from training data),
    draw posterior samples for each, and evaluate calibration and bias.

    Args:
        n_samples:           Number of out-of-sample observations to evaluate
        model_dir:           Trained run directory (relative to results/models/)
        n_posterior_samples: Posterior draws per observation
        seed:                Random seed (different from training seed=42)
        output_dir:          Where to save results; defaults to
                             results/oos/<run-name>/
    """
    model_dir = resolve(model_dir, MODELS_DIR)
    if output_dir is None:
        output_dir = RESULTS_DIR / 'oos' / model_dir.name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("STEP 03 — OUT-OF-SAMPLE VALIDATION, ACE NPE MODEL")
    print("="*70)
    print(f"  Samples:            {n_samples}")
    print(f"  Posterior draws:    {n_posterior_samples}")
    print(f"  Seed:               {seed}  (different from training seed)")
    print(f"  Model dir:          {model_dir}")
    print(f"  Output dir:         {output_dir}")

    # ---- Load posterior ----
    loaded         = load_posterior(model_dir)
    posterior      = loaded['posterior']
    feature_scaler = loaded['scaler']
    config         = loaded['config']
    param_names    = loaded['param_names']
    feature_cols   = loaded['feature_cols']
    print(f"\n✓ Loaded posterior — features: {feature_cols}")
    print(f"✓ Predicting {len(param_names)} parameters: {param_names}")

    # ---- Extract prior bounds for reference lines on plots ----
    prior_lower = np.array(config['prior_lower']) if 'prior_lower' in config else None
    prior_upper = np.array(config['prior_upper']) if 'prior_upper' in config else None

    # ---- Generate fresh OOS data ----
    print(f"\nGenerating {n_samples} out-of-sample observations...")
    df_oos = generate_training_data(n_samples=n_samples, seed=seed)

    X_oos  = df_oos[feature_cols].values          # (n, n_features)
    y_true = df_oos[param_names].values           # (n, 3)

    X_scaled = feature_scaler.transform(X_oos)

    # ---- Draw posterior samples ----
    print(f"Drawing {n_posterior_samples} posterior samples per observation...")
    pred_means, pred_stds, pred_ci_lo, pred_ci_hi, pred_maps = [], [], [], [], []

    for i in range(n_samples):
        x_obs = torch.FloatTensor(X_scaled[i]).unsqueeze(0)
        with torch.no_grad():
            samples = posterior.sample(
                (n_posterior_samples,), x=x_obs, show_progress_bars=False
            )
        s = samples.numpy()
        pred_means.append(s.mean(0))
        pred_stds.append(s.std(0))
        pred_ci_lo.append(np.percentile(s, 2.5,  axis=0))
        pred_ci_hi.append(np.percentile(s, 97.5, axis=0))
        pred_maps.append(map_from_samples(s))

    pred_means = np.array(pred_means)   # (n, 3)
    pred_stds  = np.array(pred_stds)
    pred_ci_lo = np.array(pred_ci_lo)
    pred_ci_hi = np.array(pred_ci_hi)
    pred_maps  = np.array(pred_maps)    # (n, 3)

    # ---- Bias & calibration summary ----
    print("\n" + "="*70)
    print("BIAS & CALIBRATION ANALYSIS")
    print("="*70)
    print(f"\n{'Parameter':<12} {'Mean True':<12} {'Mean Pred':<12} "
          f"{'Bias':<10} {'|B|/SD':<10} {'R²':<8} {'RMSE':<8} "
          f"{'Mean σ':<10} {'95% Cov':<8}  {'MAP Bias':<12} {'MAP R²'}")
    print("-"*112)

    results_rows = []
    for i, name in enumerate(param_names):
        true_i  = y_true[:, i]
        pred_i  = pred_means[:, i]
        map_i   = pred_maps[:, i]
        std_i   = pred_stds[:, i]
        ci_lo_i = pred_ci_lo[:, i]
        ci_hi_i = pred_ci_hi[:, i]

        bias      = float(np.mean(pred_i - true_i))
        map_bias  = float(np.mean(map_i  - true_i))
        sd        = float(np.std(true_i))
        rel_bias  = abs(bias) / sd if sd > 0 else float('nan')
        r2        = float(r2_score(true_i, pred_i))
        r2_map    = float(r2_score(true_i, map_i))
        rmse      = float(np.sqrt(mean_squared_error(true_i, pred_i)))
        mae       = float(mean_absolute_error(true_i, pred_i))
        mean_sig  = float(std_i.mean())
        coverage  = float(np.mean((true_i >= ci_lo_i) & (true_i <= ci_hi_i)))

        print(f"{name:<12} {np.mean(true_i):<12.4f} {np.mean(pred_i):<12.4f} "
              f"{bias:<10.4f} {rel_bias:<10.4f} {r2:<8.4f} {rmse:<8.4f} "
              f"{mean_sig:<10.4f} {coverage:<8.3f}  {map_bias:<+12.4f} {r2_map:.4f}")

        results_rows.append({
            'param':      name,
            'mean_true':  float(np.mean(true_i)),
            'mean_pred':  float(np.mean(pred_i)),
            'bias':       bias,
            'map_bias':   map_bias,
            'rel_bias':   rel_bias,
            'r2':         r2,
            'r2_map':     r2_map,
            'rmse':       rmse,
            'mae':        mae,
            'mean_sigma': mean_sig,
            'coverage_95': coverage,
        })

    print("="*70)
    print("  bias      = mean(posterior mean) - mean(true)")
    print("  |B|/SD    = |bias| / SD(true)  (< 0.1 is good)")
    print("  95% Cov   = fraction of true values inside 95% posterior CI  (target ≈ 0.95)")

    # ---- Save detailed predictions ----
    df_out_cols = (
        [f"{p}_true"    for p in param_names] +
        [f"{p}_pred"    for p in param_names] +
        [f"{p}_map"     for p in param_names] +
        [f"{p}_std"     for p in param_names] +
        [f"{p}_ci_lo"   for p in param_names] +
        [f"{p}_ci_hi"   for p in param_names]
    )
    df_out = pd.DataFrame(
        np.hstack([y_true, pred_means, pred_maps, pred_stds, pred_ci_lo, pred_ci_hi]),
        columns=df_out_cols,
    )
    df_out['sum_true'] = y_true.sum(axis=1)
    df_out['sum_pred'] = pred_means.sum(axis=1)

    predictions_path = output_dir / 'oos_predictions.csv'
    df_out.to_csv(predictions_path, index=False)
    print(f"\n✓ Detailed predictions saved to: {predictions_path}")

    summary_df = pd.DataFrame(results_rows)
    summary_df.to_csv(output_dir / 'oos_bias_summary.csv', index=False)
    print(f"✓ Bias summary saved to: {output_dir / 'oos_bias_summary.csv'}")

    # ---- Plots ----
    _plot_oos(y_true, pred_means, pred_stds, param_names, output_dir,
              pred_maps=pred_maps, prior_lower=prior_lower, prior_upper=prior_upper)

    return df_out


# ============================================================================
# VISUALIZATION
# ============================================================================

def _plot_oos(y_true, pred_means, pred_stds, param_names, output_dir,
              pred_maps=None, prior_lower=None, prior_upper=None):
    """Scatter plots with error bars, bias distribution histograms, and MAP scatter."""
    n      = len(param_names)
    n_cols = min(n, 4)
    n_rows = math.ceil(n / n_cols)

    # --- Posterior mean vs True ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    for i, name in enumerate(param_names):
        ax   = axes[i]
        t    = y_true[:, i]
        p    = pred_means[:, i]
        s    = pred_stds[:, i]
        r2   = r2_score(t, p)
        bias = np.mean(p - t)

        ax.errorbar(t, p, yerr=s, fmt='o', alpha=0.4, markersize=4,
                    elinewidth=0.6, capsize=2)
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lims, lims, 'r--', lw=2, label='Identity')
        if prior_lower is not None and prior_upper is not None:
            pm = 0.5 * (prior_lower[i] + prior_upper[i])
            ax.axhline(pm, color='purple', linestyle=':', lw=1.5,
                       label=f'Prior mean = {pm:.3f}')
            ax.axvline(pm, color='purple', linestyle=':', lw=1.5)
        ax.set_xlabel('True Value', fontsize=10)
        ax.set_ylabel('Posterior Mean', fontsize=10)
        ax.set_title(f'{name}  (R²={r2:.3f}, bias={bias:+.4f})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.suptitle('ACE NPE — Out-of-Sample: Posterior Mean vs True  (error bars = posterior σ)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'oos_predictions_vs_true.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved predictions vs true plot")

    # --- Bias distribution (prediction error histogram) ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, name in enumerate(param_names):
        ax    = axes[i]
        error = pred_means[:, i] - y_true[:, i]
        ax.hist(error, bins=30, edgecolor='k', alpha=0.7)
        ax.axvline(0,              color='r',      lw=2, linestyle='--', label='Zero bias')
        ax.axvline(np.mean(error), color='orange', lw=2, linestyle='-',
                   label=f'Mean = {np.mean(error):+.4f}')
        ax.set_xlabel('Posterior Mean − True', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.suptitle('ACE NPE — Prediction Error Distribution (Out-of-Sample)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'oos_bias_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved bias distribution plot")

    # --- Posterior width (σ) per observation ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, name in enumerate(param_names):
        ax = axes[i]
        ax.hist(pred_stds[:, i], bins=30, edgecolor='k', alpha=0.7, color='steelblue')
        ax.axvline(pred_stds[:, i].mean(), color='r', lw=2, linestyle='--',
                   label=f'Mean σ = {pred_stds[:, i].mean():.4f}')
        ax.set_xlabel('Posterior σ', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.suptitle('ACE NPE — Posterior Width Distribution (Out-of-Sample)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'oos_posterior_width.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved posterior width plot")

    # --- MAP estimate vs True ---
    if pred_maps is not None:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = np.array(axes).flatten()

        for i, name in enumerate(param_names):
            ax     = axes[i]
            t      = y_true[:, i]
            m      = pred_maps[:, i]
            r2_m   = r2_score(t, m)
            bias_m = np.mean(m - t)

            ax.scatter(t, m, alpha=0.4, s=15, color='steelblue')
            lims = [min(t.min(), m.min()), max(t.max(), m.max())]
            ax.plot(lims, lims, 'r--', lw=2, label='Identity')
            if prior_lower is not None and prior_upper is not None:
                pm = 0.5 * (prior_lower[i] + prior_upper[i])
                ax.axhline(pm, color='purple', linestyle=':', lw=1.5,
                           label=f'Prior mean = {pm:.3f}')
                ax.axvline(pm, color='purple', linestyle=':', lw=1.5)
            ax.set_xlabel('True Value', fontsize=10)
            ax.set_ylabel('MAP Estimate', fontsize=10)
            ax.set_title(f'{name}  MAP  (R²={r2_m:.3f}, bias={bias_m:+.4f})',
                         fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for j in range(n, n_rows * n_cols):
            fig.delaxes(axes[j])

        plt.suptitle('ACE NPE — Out-of-Sample: MAP Estimate vs True',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'oos_map_vs_true.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved MAP vs true plot")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Out-of-sample bias validation for the ACE NPE model'
    )
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Number of out-of-sample observations (default: 200)')
    parser.add_argument('--model_dir', type=str, default=str(DEFAULT_MODEL_DIR),
                        help='Trained run directory, relative to results/models/ '
                             f'(default: {DEFAULT_MODEL_DIR.name})')
    parser.add_argument('--n_posterior_samples', type=int, default=500,
                        help='Posterior draws per observation (default: 500)')
    parser.add_argument('--seed', type=int, default=999,
                        help='Random seed for OOS data generation (default: 999)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to write results (default: results/oos/<run-name>/)')
    args = parser.parse_args()

    run_oos_validation(
        n_samples=args.n_samples,
        model_dir=args.model_dir,
        n_posterior_samples=args.n_posterior_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )
