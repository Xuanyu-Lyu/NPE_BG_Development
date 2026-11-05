"""Integration check for the fixed-total OpenMx ACE backend."""

import csv
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1] / "04_fit_openmx_paired_dirichlet.R"


class FixedTotalOpenMxCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("Rscript") is None:
            raise unittest.SkipTest("Rscript is unavailable")
        available = subprocess.run(
            ["Rscript", "-e", "quit(status=!requireNamespace('OpenMx', quietly=TRUE))"],
            capture_output=True,
            text=True,
            check=False,
        )
        if available.returncode != 0:
            raise unittest.SkipTest("OpenMx is unavailable")

    def test_estimates_sum_to_one_and_boundary_intervals_are_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "paired.csv"
            output_path = Path(directory) / "openmx.csv"
            rows = []
            for index, (a, c, e) in enumerate(
                ((0.4, 0.3, 0.3), (0.0, 0.4, 0.6), (0.4, 0.0, 0.6)), 1
            ):
                rows.append({
                    "condition_id": index, "N_pairs": 2000,
                    "A_true": a, "C_true": c, "E_true": e,
                    "mz_var1": 1.0, "mz_cov": a + c, "mz_var2": 1.0,
                    "dz_var1": 1.0, "dz_cov": 0.5 * a + c, "dz_var2": 1.0,
                })
            with input_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            subprocess.run(
                ["Rscript", str(BACKEND), str(input_path), str(output_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            with output_path.open(newline="") as handle:
                fits = list(csv.DictReader(handle))

        self.assertEqual(len(fits), 3)
        for truth, fit in zip(rows, fits):
            self.assertEqual(fit["openmx_model"], "fixed_total_variance_1")
            self.assertEqual(int(fit["openmx_converged"]), 1)
            estimates = [float(fit[f"{p}_openmx_est"]) for p in "ACE"]
            self.assertAlmostEqual(sum(estimates), 1.0, places=10)
            for parameter, estimate in zip("ACE", estimates):
                self.assertAlmostEqual(
                    estimate, truth[f"{parameter}_true"], delta=0.02
                )
                self.assertTrue(math.isfinite(float(fit[f"{parameter}_openmx_se"])))
            for parameter in "AC":
                lower = float(fit[f"{parameter}_openmx_ci_lo"])
                upper = float(fit[f"{parameter}_openmx_ci_hi"])
                estimate = float(fit[f"{parameter}_openmx_est"])
                self.assertLessEqual(lower, estimate + 1e-10)
                self.assertLessEqual(estimate, upper + 1e-10)


if __name__ == "__main__":
    unittest.main()
