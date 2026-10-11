"""Check median summaries, valid uncertainties, and dataset-level pairing."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openmx_comparison_plots import (
    fixed_n_estimates_long,
    paired_scatter_data,
    plot_paired_scatter,
    plot_summary_with_boxplots,
    valid_metric_values,
)


def example_estimates():
    rows = []
    for n in (50, 20000):
        for parameter in ("A", "C", "E"):
            for method, scale in (("OpenMx", 1), ("NPE-100k", 2)):
                for dataset, sd in enumerate((0.01, 0.02, 0.03, 0.80), start=1):
                    error = scale * (dataset - 2) * 0.01
                    rows.append({
                        "N": n, "test_dataset_m": dataset,
                        "parameter": parameter, "method": method,
                        "truth": 0.3, "estimate": 0.3 + error,
                        "uncertainty": sd * scale,
                        "error": error, "absolute_error": abs(error),
                    })
    return pd.DataFrame(rows)


class ComparisonPlotChecks(unittest.TestCase):
    def test_pairing_uses_dataset_n_and_parameter_keys(self):
        estimates = example_estimates().sample(frac=1, random_state=17)
        paired = paired_scatter_data(estimates, "NPE-100k", "error")
        self.assertEqual(len(paired), 24)
        self.assertFalse(paired.duplicated(["N", "test_dataset_m", "parameter"]).any())
        np.testing.assert_allclose(paired["error_npe"], 2 * paired["error_openmx"])
        expected = (paired["test_dataset_m"] - 2) * 0.01
        np.testing.assert_allclose(paired["error_openmx"], expected)

    def test_invalid_sd_removed_jointly_and_zero_sd_retained(self):
        estimates = example_estimates()
        at_panel = estimates["N"].eq(50) & estimates["parameter"].eq("A")
        estimates.loc[at_panel & estimates["test_dataset_m"].eq(1), "uncertainty"] = 0
        estimates.loc[at_panel & estimates["test_dataset_m"].eq(2) & estimates["method"].eq("OpenMx"), "uncertainty"] = np.nan
        estimates.loc[at_panel & estimates["test_dataset_m"].eq(3) & estimates["method"].eq("NPE-100k"), "uncertainty"] = -0.1
        estimates.loc[at_panel & estimates["test_dataset_m"].eq(4) & estimates["method"].eq("NPE-100k"), "estimate"] = np.inf
        paired = paired_scatter_data(estimates, "NPE-100k", "uncertainty")
        panel = paired.loc[paired["N"].eq(50) & paired["parameter"].eq("A")]
        self.assertEqual(panel["test_dataset_m"].tolist(), [1])
        errors = paired_scatter_data(estimates, "NPE-100k", "error")
        panel = errors.loc[errors["N"].eq(50) & errors["parameter"].eq("A")]
        self.assertEqual(panel["test_dataset_m"].tolist(), [1, 2, 3])

    def test_rejects_duplicate_pairs_and_inconsistent_truth(self):
        estimates = example_estimates()
        with self.assertRaises(pd.errors.MergeError):
            paired_scatter_data(pd.concat([estimates, estimates.iloc[:1]]), "NPE-100k", "error")
        estimates.loc[estimates["method"].eq("NPE-100k"), "truth"] = 0.4
        with self.assertRaisesRegex(ValueError, "ground truth"):
            paired_scatter_data(estimates, "NPE-100k", "error")

    def test_summary_uses_median_and_boxes_use_the_test_distribution(self):
        estimates = example_estimates()
        panel = estimates.loc[estimates["N"].eq(50) & estimates["parameter"].eq("A")]
        values = valid_metric_values(panel.loc[panel["method"].eq("OpenMx")], "uncertainty")
        self.assertNotAlmostEqual(values.mean(), np.median(values))
        def check_plot(figure, *args, **kwargs):
            heights = [bar.get_height() for bar in figure.axes[0].patches]
            np.testing.assert_allclose(heights, [0.025, 0.05])
            self.assertEqual(len(figure.axes), 12)  # two N ranges, summary + boxes
            self.assertEqual(len(figure.axes[3].patches), 2)
        with patch("matplotlib.figure.Figure.savefig", autospec=True, side_effect=check_plot):
            plot_summary_with_boxplots(
                estimates, ("OpenMx", "NPE-100k"), (50, 20000),
                {"OpenMx": "blue", "NPE-100k": "orange"},
                "uncertainty", "Median uncertainty", Path("unused.png"),
            )

    def test_fixed_n_adapter_excludes_unconverged_and_nonfinite_estimates(self):
        wide = pd.DataFrame({
            "N_pairs": [50, 50, 50], "condition_id": [1, 2, 3],
            "openmx_converged": [1, 0, 1],
        })
        for parameter in ("A", "C", "E"):
            wide[f"{parameter}_true"] = 0.3
            wide[f"{parameter}_openmx_est"] = [0.4, 0.5, np.nan]
            wide[f"{parameter}_openmx_se"] = 0.1
            wide[f"{parameter}_npe_mean"] = 0.35
            wide[f"{parameter}_npe_sd"] = 0.08
        output = fixed_n_estimates_long(wide)
        self.assertEqual(len(output), 6)
        self.assertEqual(set(output["test_dataset_m"]), {1})
        np.testing.assert_allclose(output.loc[output["method"].eq("OpenMx"), "error"], 0.1)

    def test_scatter_writes_both_types_including_empty_panel(self):
        estimates = example_estimates()
        estimates.loc[estimates["parameter"].eq("E"), "uncertainty"] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            for metric in ("uncertainty", "error"):
                path = Path(directory) / f"{metric}.png"
                plot_paired_scatter(estimates, "NPE-100k", (50,), metric, "Paired comparison", path)
                self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
