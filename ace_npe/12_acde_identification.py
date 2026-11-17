"""STEP 12: ACDE nonidentification under a uniform Dirichlet prior.

Run one fixed-N cell per task (``--cell-index 1`` through ``7``), then
``--aggregate``. ``--run-all`` runs the cells sequentially. Defaults are
K=200,000, M=1,000, L=2,000 and the Step 06 N grid. The two focused
datasets use N=1,000 and are evaluated only in that cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from torch.distributions import Distribution, constraints

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_model import COV_FEATURE_NAMES, RESULTS_DIR, VAR_REDUCTION, resolve
from posterior_predictive_checks import summarize_ppc
from training_budget_utils import (
    _wishart_compound_symmetric_summaries,
    resolve_device,
    seed_from,
)

PARAMS = ("A", "C", "D", "E")
N_VALUES = (50, 100, 500, 1_000, 2_000, 5_000, 20_000)
FOCUSED = (
    ("case_1", (0.30, 0.30, 0.20, 0.20)),
    ("case_2", (0.45, 0.25, 0.10, 0.20)),
)


def population_features(acde: np.ndarray) -> np.ndarray:
    """Population (mz_var, mz_cov, dz_var, dz_cov), with total variance 1."""
    theta = np.asarray(acde, dtype=float)
    if theta.shape[-1] != 4:
        raise ValueError("ACDE must have four columns")
    a, c, d, _e = np.moveaxis(theta, -1, 0)
    one = np.ones_like(a)
    return np.stack((one, a + c + d, one, 0.5 * a + c + 0.25 * d), axis=-1)


def limiting_posterior_sd(acde: np.ndarray) -> np.ndarray:
    """Analytic N->infinity marginal SD for a uniform prior on the ACDE simplex.

    Conditioning the uniform prior on exact MZ/DZ population covariances
    leaves D uniform on its feasible interval. A and C are linear in D;
    E is identified. The result is ordered A, C, D, E.
    """
    features = population_features(acde)
    s, q = features[..., 1], features[..., 3]
    lower = np.maximum(0.0, 2.0 * (s - 2.0 * q))
    upper = (4.0 / 3.0) * (s - q)
    width = np.maximum(0.0, upper - lower)
    return width[..., None] / np.sqrt(12.0) * np.array([1.5, 0.5, 1.0, 0.0])


def simulate_acde(acde: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Exact Gaussian sample-covariance summaries via Wishart, df=N-1."""
    theta = np.asarray(acde, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 4 or len(theta) == 0:
        raise ValueError("ACDE must have shape (rows, 4)")
    if (not np.isfinite(theta).all() or np.any(theta <= 0)
            or not np.allclose(theta.sum(axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError("ACDE rows must be positive and sum to one")
    theta = theta / theta.sum(axis=1, keepdims=True)
    rho_mz = theta[:, 0] + theta[:, 1] + theta[:, 2]
    rho_dz = 0.5 * theta[:, 0] + theta[:, 1] + 0.25 * theta[:, 2]
    mz = _wishart_compound_symmetric_summaries(rho_mz, n, seed_from(seed, 1))
    dz = _wishart_compound_symmetric_summaries(rho_dz, n, seed_from(seed, 2))
    return np.column_stack((0.5 * (mz[0] + mz[2]), mz[1],
                            0.5 * (dz[0] + dz[2]), dz[1])).astype(np.float32)


class DirichletACDEALRPrior(Distribution):
    """Dirichlet(1,1,1,1) in log(A/E), log(C/E), log(D/E) coordinates."""

    arg_constraints = {}
    support = constraints.independent(constraints.real, 1)
    has_rsample = False

    def __init__(self, device="cpu", validate_args=None):
        self.alpha = torch.ones(4, device=device)
        super().__init__(torch.Size(), torch.Size([3]), validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        p = torch.distributions.Dirichlet(self.alpha).sample(torch.Size(sample_shape))
        return torch.log(p[..., :3] / p[..., 3:4])

    def log_prob(self, value):
        zero = torch.zeros((*value.shape[:-1], 1), device=value.device,
                           dtype=value.dtype)
        p = torch.softmax(torch.cat((value, zero), dim=-1), dim=-1)
        # Inverse additive-log-ratio Jacobian on the simplex is prod(p_i).
        return (torch.distributions.Dirichlet(self.alpha.to(value)).log_prob(p)
                + torch.log(p).sum(dim=-1))

    def to(self, device):
        self.alpha = self.alpha.to(device)
        return self


def train(features: np.ndarray, theta: np.ndarray, args):
    n_scaler = max(2, int((1 - args.validation_fraction) * len(features)))
    scaler = StandardScaler().fit(features[:n_scaler])
    x = scaler.transform(features).astype(np.float32)
    latent = np.log(theta[:, :3].astype(np.float64) /
                    theta[:, 3:4].astype(np.float64)).astype(np.float32)
    torch.manual_seed(args.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.training_seed)
    builder = posterior_nn(
        model=args.flow_type, embedding_net=nn.Identity(),
        hidden_features=args.flow_hidden, num_transforms=args.flow_transforms,
        z_score_theta="independent", z_score_x="independent",
    )
    inference = SNPE(prior=DirichletACDEALRPrior(str(args.device_resolved)),
                     density_estimator=builder, device=str(args.device_resolved))
    inference.append_simulations(torch.from_numpy(latent), torch.from_numpy(x))
    density = inference.train(
        training_batch_size=args.batch_size, learning_rate=args.learning_rate,
        max_num_epochs=args.epochs, stop_after_epochs=args.stop_after_epochs,
        validation_fraction=args.validation_fraction, show_train_summary=True,
    )
    posterior = inference.build_posterior(density)
    posterior.to("cpu")
    return posterior, scaler


def sample_posterior(posterior, scaler, feature, draws: int) -> np.ndarray:
    x = torch.from_numpy(scaler.transform(np.asarray(feature)[None, :]).astype(np.float32))
    with torch.no_grad():
        latent = posterior.sample((draws,), x=x, show_progress_bars=False)
        zero = torch.zeros((*latent.shape[:-1], 1), dtype=latent.dtype)
        samples = torch.softmax(torch.cat((latent, zero), dim=-1), dim=-1)
    return samples.cpu().numpy()


def focus_path(output: Path) -> Path:
    return output / "cells" / "N1000_focused.npz"


def cell_metadata(n: int, args) -> dict:
    return {
        "step": 12, "K": args.k, "N": n, "M": args.m, "L": args.l,
        "prior": "Dirichlet(1,1,1,1)", "theta_order": PARAMS,
        "theta_coordinates": ["log(A/E)", "log(C/E)", "log(D/E)"],
        "features": COV_FEATURE_NAMES, "var_feature": VAR_REDUCTION,
        "simulator": "Gaussian Wishart sample covariance, df=N-1",
        "estimate": "joint posterior draw mean", "seed": args.seed,
        "training_seed": args.training_seed,
        "focused_n": args.focus_n, "ppc_replicates": args.ppc_replicates,
        "training": {key: getattr(args, key) for key in (
            "validation_fraction", "epochs", "stop_after_epochs", "batch_size",
            "learning_rate", "flow_type", "flow_hidden", "flow_transforms")},
    }


def run_cell(n: int, args, output: Path) -> None:
    cells = output / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    result_path = cells / f"N{n}.csv"
    config_path = cells / f"N{n}.json"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(f"{result_path} exists; use --overwrite to rerun")
    print(f"Step 12: K={args.k:,}, N={n:,}, M={args.m:,}, L={args.l:,}", flush=True)
    train_rng = np.random.default_rng(seed_from(args.seed, n, 1))
    theta_train = train_rng.dirichlet(np.ones(4), size=args.k).astype(np.float32)
    x_train = simulate_acde(theta_train, n, seed_from(args.seed, n, 2))
    posterior, scaler = train(x_train, theta_train, args)
    del x_train, theta_train

    # Identical test truths across all N cells; data noise changes with N.
    theta_test = np.random.default_rng(seed_from(args.seed, 3)).dirichlet(
        np.ones(4), size=args.m).astype(np.float32)
    limiting_sd = limiting_posterior_sd(theta_test)
    x_test = simulate_acde(theta_test, n, seed_from(args.seed, n, 4))
    torch.manual_seed(seed_from(args.seed, n, 5))
    rows = []
    for m, (truth, observed) in enumerate(zip(theta_test, x_test), start=1):
        draws = sample_posterior(posterior, scaler, observed, args.l)
        means = draws.mean(axis=0)
        sds = draws.std(axis=0, ddof=1)
        for j, param in enumerate(PARAMS):
            rows.append({"K": args.k, "N": n, "M": m, "parameter": param,
                         "truth": float(truth[j]), "estimate": float(means[j]),
                         "posterior_sd": float(sds[j]),
                         "identification_limit_sd": float(limiting_sd[m - 1, j]),
                         "absolute_error": float(abs(means[j] - truth[j]))})
        if m % 100 == 0:
            print(f"  evaluated {m}/{args.m}", flush=True)
    temporary_result = result_path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(temporary_result, index=False)
    temporary_result.replace(result_path)

    if n == args.focus_n:
        focused_theta = np.asarray([theta for _, theta in FOCUSED], dtype=np.float32)
        observed = simulate_acde(focused_theta, n, seed_from(args.seed, 6))
        focus_draws = np.stack([
            sample_posterior(posterior, scaler, obs, args.l) for obs in observed
        ])
        ppc = np.stack([
            simulate_acde(draws[np.random.default_rng(seed_from(args.seed, i, 7))
                                .choice(args.l, size=args.ppc_replicates,
                                        replace=False)], n, seed_from(args.seed, i, 8))
            for i, draws in enumerate(focus_draws)
        ])
        focused_file = focus_path(output)
        temporary_focused = focused_file.with_suffix(".npz.tmp")
        with temporary_focused.open("wb") as handle:
            np.savez_compressed(handle, truths=focused_theta, observed=observed,
                                posterior=focus_draws, ppc=ppc)
        temporary_focused.replace(focused_file)

    temporary_config = config_path.with_suffix(".json.tmp")
    temporary_config.write_text(json.dumps(cell_metadata(n, args), indent=2))
    temporary_config.replace(config_path)
    print(f"Saved {result_path}", flush=True)


def save_boxplots(frame: pd.DataFrame, n_values: tuple[int, ...], column: str,
                  path: Path, args) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for j, param in enumerate(PARAMS):
        ax = axes.flat[j]
        data = [frame.loc[(frame.N == n) & (frame.parameter == param), column].to_numpy()
                for n in n_values]
        ax.boxplot(data, positions=np.arange(len(n_values)), widths=0.6,
                   showfliers=False, medianprops={"color": "#CC3311"})
        ax.set_title(param)
        ax.set_ylabel(column.replace("_", " ").title())
        ax.set_xticks(range(len(n_values)), [str(n) for n in n_values], rotation=45)
        ax.set_xlabel("Twin pairs per zygosity group (N)")
    fig.suptitle(f"ACDE {column.replace('_', ' ')}; "
                 f"K={args.k:,}; M={args.m:,}; L={args.l:,}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_scatter(frame: pd.DataFrame, n: int, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for j, param in enumerate(PARAMS):
        ax = axes.flat[j]
        sub = frame.loc[(frame.N == n) & (frame.parameter == param)]
        ax.scatter(sub.truth, sub.estimate, s=8, alpha=0.35, color="#4477AA")
        ax.plot([0, 1], [0, 1], "--", color="#333333", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=param,
               xlabel="True parameter", ylabel="Posterior mean estimate")
    fig.suptitle(f"ACDE recovery; N={n:,}; M={len(sub):,}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_focus_corner(draws: np.ndarray, label: str, path: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(11, 11))
    d = np.linspace(0, 0.4, 201)
    ridge = np.column_stack((0.6 - 1.5 * d, 0.2 + 0.5 * d, d,
                             np.full_like(d, 0.2)))
    colors = ("#CC3311", "#228833")
    for row in range(4):
        for col in range(4):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
            elif col == row:
                ax.hist(draws[:, row], bins=35, density=True,
                        color="#4477AA", alpha=0.75)
                for i, (_, theta) in enumerate(FOCUSED):
                    ax.axvline(theta[row], color=colors[i], linestyle="--")
                ax.set(xlim=(0, 1), xlabel=PARAMS[col], ylabel="Density")
            else:
                ax.scatter(draws[:, col], draws[:, row], s=3, alpha=0.15,
                           color="#4477AA", rasterized=True)
                ax.plot(ridge[:, col], ridge[:, row], color="black", linewidth=1.5)
                for i, (_, theta) in enumerate(FOCUSED):
                    ax.plot(theta[col], theta[row], "o", color=colors[i], markersize=6)
                ax.set(xlim=(0, 1), ylim=(0, 1),
                       xlabel=PARAMS[col], ylabel=PARAMS[row])
    fig.suptitle(f"{label}: N=1,000; blue posterior; black identification line\n"
                 "red: (0.30,0.30,0.20,0.20); green: (0.45,0.25,0.10,0.20)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_focused_ppc_comparison(focused: dict, path: Path, n: int) -> None:
    """Match Step 11's two-row, four-summary-plus-joint PPC layout."""
    colors = ("#4477AA", "#228833")
    fig, axes = plt.subplots(2, 5, figsize=(19, 7.6), squeeze=False)
    observed = focused["observed"]
    replicated = focused["ppc"]
    for col, feature in enumerate(COV_FEATURE_NAMES):
        combined = np.concatenate((replicated[:, :, col].ravel(), observed[:, col]))
        bins = np.linspace(combined.min(), combined.max(), 31)
        for row, (_, theta) in enumerate(FOCUSED):
            ax = axes[row, col]
            values = replicated[row, :, col]
            low, high = np.quantile(values, [0.025, 0.975])
            ax.hist(values, bins=bins, density=True, color=colors[row], alpha=0.7)
            ax.axvspan(low, high, color=colors[row], alpha=0.10)
            ax.axvline(observed[row, col], color="black", linestyle="--", linewidth=1.6)
            ax.set_xlabel(feature)
            if col == 0:
                ax.set_ylabel(f"ACDE truth {row + 1}\nDensity")
            ax.grid(alpha=0.15)
        axes[1, col].sharex(axes[0, col])

    all_x = np.concatenate((replicated[:, :, 1].ravel(), observed[:, 1]))
    all_y = np.concatenate((replicated[:, :, 3].ravel(), observed[:, 3]))
    x_pad = max(0.01, 0.05 * np.ptp(all_x))
    y_pad = max(0.01, 0.05 * np.ptp(all_y))
    for row, _ in enumerate(FOCUSED):
        ax = axes[row, 4]
        ax.scatter(replicated[row, :, 1], replicated[row, :, 3],
                   s=12, alpha=0.3, color=colors[row])
        ax.plot(observed[row, 1], observed[row, 3], "*", color="black", markersize=13)
        ax.set(xlabel="mz_cov", ylabel="dz_cov",
               xlim=(all_x.min() - x_pad, all_x.max() + x_pad),
               ylim=(all_y.min() - y_pad, all_y.max() + y_pad))
        ax.grid(alpha=0.15)
    fig.suptitle(
        f"ACDE focused observations; N={n:,} pairs per group\n"
        "Truth 1: (0.30, 0.30, 0.20, 0.20); "
        "Truth 2: (0.45, 0.25, 0.10, 0.20)\n"
        "Dashed lines and stars: observed; shading: 95% posterior predictive interval",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def aggregate(args, output: Path) -> None:
    cells = output / "cells"
    frames = []
    for n in args.n_values:
        path = cells / f"N{n}.csv"
        meta_path = cells / f"N{n}.json"
        if not path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Complete N={n} cell before aggregating")
        meta = json.loads(meta_path.read_text())
        if meta != json.loads(json.dumps(cell_metadata(n, args))):
            raise ValueError(f"Cell settings differ: {meta_path}")
        frame = pd.read_csv(path)
        expected_pairs = pd.MultiIndex.from_product(
            [range(1, args.m + 1), PARAMS], names=["M", "parameter"])
        actual_pairs = pd.MultiIndex.from_frame(frame[["M", "parameter"]])
        if (len(frame) != 4 * args.m or not actual_pairs.is_unique
                or not actual_pairs.sort_values().equals(expected_pairs.sort_values())
                or not (frame["N"] == n).all() or not (frame["K"] == args.k).all()
                or not np.isfinite(frame[["truth", "estimate", "posterior_sd",
                                          "absolute_error", "identification_limit_sd"]]
                                   .to_numpy()).all()):
            raise ValueError(f"Incomplete cell: {path}")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output / "acde_recovery.csv", index=False)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    # An earlier Step 12 version produced this figure; remove it on re-aggregation.
    (figures / "covariance_recovery_two_cases.png").unlink(missing_ok=True)
    for col in ("posterior_sd", "absolute_error"):
        save_boxplots(result, args.n_values, col,
                      figures / f"{col}_boxplot.png", args)
    for n in args.n_values:
        save_scatter(result, n, figures / f"recovery_scatter_N{n}.png")

    if args.focus_n not in args.n_values or not focus_path(output).exists():
        raise FileNotFoundError("Focused N=1,000 cell must be present")
    with np.load(focus_path(output)) as saved:
        focused = {key: saved[key] for key in saved.files}
    if not np.allclose(population_features(focused["truths"])[0],
                       population_features(focused["truths"])[1], atol=1e-8):
        raise ValueError("Focused cases do not share population covariances")
    save_focused_ppc_comparison(focused, figures / "ppc_two_cases.png", args.focus_n)
    ppc_rows = []
    covariance_rows = []
    for i, (label, _) in enumerate(FOCUSED):
        draws = focused["posterior"][i]
        observed = focused["observed"][i]
        truth = focused["truths"][i]
        save_focus_corner(draws, label, figures / f"posterior_{label}.png")
        for row in summarize_ppc(focused["ppc"][i], observed):
            ppc_rows.append({"case": label, **row})
        population_draws = population_features(draws)
        population_truth = population_features(truth)
        for j, feature in enumerate(COV_FEATURE_NAMES):
            covariance_rows.append({
                "case": label, "feature": feature,
                "observed": float(observed[j]),
                "population_truth": float(population_truth[j]),
                "posterior_prediction_mean": float(population_draws[:, j].mean()),
                "posterior_prediction_q2_5": float(np.quantile(population_draws[:, j], 0.025)),
                "posterior_prediction_q97_5": float(np.quantile(population_draws[:, j], 0.975)),
            })
    pd.DataFrame(ppc_rows).to_csv(output / "focused_ppc.csv", index=False)
    pd.DataFrame(covariance_rows).to_csv(output / "focused_covariance_recovery.csv",
                                         index=False)
    print(f"Saved results and figures to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cell-index", type=int, help="One-based index into N grid")
    mode.add_argument("--aggregate", action="store_true")
    mode.add_argument("--run-all", action="store_true")
    parser.add_argument("--k", type=int, default=200_000)
    parser.add_argument("--n-values", nargs="+", type=int, default=N_VALUES)
    parser.add_argument("--focus-n", type=int, default=1_000)
    parser.add_argument("--m", type=int, default=1_000)
    parser.add_argument("--l", type=int, default=2_000)
    parser.add_argument("--ppc-replicates", type=int, default=500)
    parser.add_argument("--output-dir", default="step12_acde_identification")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--stop-after-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--flow-type", choices=("nsf", "maf", "maf_rqs", "mdn"), default="nsf")
    parser.add_argument("--flow-hidden", type=int, default=64)
    parser.add_argument("--flow-transforms", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.n_values = tuple(dict.fromkeys(args.n_values))
    if args.k < 100 or args.m < 2 or args.l < 20 or min(args.n_values) < 2:
        parser.error("K>=100, M>=2, L>=20, and N>=2 are required")
    if not 2 <= args.ppc_replicates <= args.l:
        parser.error("PPC replicates must be between 2 and L")
    if args.focus_n != 1000 or args.focus_n not in args.n_values:
        parser.error("The agreed focused datasets require N=1000 in --n-values")
    if not 0 < args.validation_fraction < 0.5:
        parser.error("validation fraction must lie between 0 and 0.5")
    if min(args.epochs, args.stop_after_epochs, args.batch_size) < 1:
        parser.error("training counts must be positive")
    if args.cell_index is not None and not 1 <= args.cell_index <= len(args.n_values):
        parser.error("cell index is outside N grid")
    args.device_resolved = resolve_device(args.device)
    return args


def main():
    args = parse_args()
    output = resolve(args.output_dir, RESULTS_DIR)
    output.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate(args, output)
    elif args.run_all:
        for n in args.n_values:
            run_cell(n, args, output)
        aggregate(args, output)
    else:
        run_cell(args.n_values[args.cell_index - 1], args, output)


if __name__ == "__main__":
    main()
