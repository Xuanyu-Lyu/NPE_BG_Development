"""Single-dataset posterior inspection, using the joint draws from SBC."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ace_model import ACE_PARAM_NAMES


def save_corner_plot(draws: np.ndarray, truth: np.ndarray, path: Path, title: str) -> None:
    """Plot one posterior's marginals and pairwise draws on the ACE simplex.

    Scatter plots avoid a three-dimensional KDE: A+C+E=1 makes that joint
    density singular in three-dimensional Euclidean space.
    """
    limits = []
    for index in range(3):
        low = min(float(draws[:, index].min()), float(truth[index]))
        high = max(float(draws[:, index].max()), float(truth[index]))
        margin = 0.08 * max(high - low, 0.005)
        limits.append((max(0, low - margin), min(1, high + margin)))
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue
            if row == col:
                ax.hist(draws[:, row], bins=35, density=True, color="#4477AA", alpha=0.75)
                lo, hi = np.quantile(draws[:, row], [0.025, 0.975])
                ax.axvspan(lo, hi, color="#4477AA", alpha=0.12)
                ax.axvline(truth[row], color="#CC3311", linestyle="--", label="Truth")
                ax.set_ylabel("Density")
                if row == 0:
                    ax.legend(frameon=False)
            else:
                ax.scatter(draws[:, col], draws[:, row], s=4, alpha=0.15,
                           color="#4477AA", rasterized=True)
                ax.axvline(truth[col], color="#CC3311", linestyle="--", linewidth=1)
                ax.axhline(truth[row], color="#CC3311", linestyle="--", linewidth=1)
                ax.plot(truth[col], truth[row], "+", color="#CC3311", markersize=10)
                ax.set_ylabel(ACE_PARAM_NAMES[row])
            ax.set_xlim(*limits[col])
            if row != col:
                ax.set_ylim(*limits[row])
            ax.set_xlabel(ACE_PARAM_NAMES[col])
    fig.suptitle(title + "\nJoint posterior draws; shaded marginal intervals: 95%", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
