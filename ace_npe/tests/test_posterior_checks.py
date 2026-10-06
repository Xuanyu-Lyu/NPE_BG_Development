"""Statistical and workflow checks for posterior inspection and noisy PPCs."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from posterior_predictive_checks import posterior_predictive_replicates, summarize_ppc
from training_budget_utils import evaluate_cell, simulate_fixed_n, simulate_summaries_given_ace

spec = importlib.util.spec_from_file_location(
    "npe_diagnostics_ppc", Path(__file__).resolve().parents[1] / "06_npe_diagnostics.py"
)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class Posterior:
    def sample(self, shape, **kwargs):
        return torch.softmax(torch.randn(shape[0], 3), dim=-1)


class PosteriorChecks(unittest.TestCase):
    def test_conditional_simulator_matches_gaussian_sample_covariance_moments(self):
        # Independent analytic Wishart moments, including covariance between
        # the mean diagonal and off-diagonal entries.
        n = 50
        draws = np.tile([0.4, 0.3, 0.3], (80000, 1))
        simulated = simulate_summaries_given_ace(draws, n, 2026)
        np.testing.assert_allclose(simulated.mean(axis=0), [1, 0.7, 1, 0.5], atol=0.003)
        expected_var = np.array([1 + 0.7**2] * 2 + [1 + 0.5**2] * 2) / (n - 1)
        np.testing.assert_allclose(simulated.var(axis=0), expected_var, rtol=0.025)
        self.assertAlmostEqual(np.cov(simulated[:, :2], rowvar=False)[0, 1], 2 * 0.7 / (n - 1), delta=0.001)
        self.assertAlmostEqual(np.cov(simulated[:, [1, 3]], rowvar=False)[0, 1], 0, delta=0.001)
        self.assertTrue(np.any(simulated[:, 0] != 1))  # genuine sampling noise

    def test_n_two_and_invalid_parameters(self):
        replicated = simulate_summaries_given_ace(np.tile([0.4, 0.3, 0.3], (100, 1)), 2, 42)
        self.assertTrue(np.isfinite(replicated).all())
        for ace in ([[0.4, 0.3, 0.4]], [[0.4, np.nan, 0.6]], [[-0.1, 0.4, 0.7]], [[1, 0, 0]]):
            with self.subTest(ace=ace), self.assertRaises(ValueError):
                simulate_summaries_given_ace(ace, 100, 42)

    def test_replicates_reproducible_without_mutating_draws_or_rng(self):
        draws = np.random.default_rng(20).dirichlet(np.ones(3), size=100)
        original = draws.copy()
        np.random.seed(10)
        torch.manual_seed(10)
        expected_numpy = np.random.rand(3)
        expected_torch = torch.rand(3)
        np.random.seed(10)
        torch.manual_seed(10)
        first = posterior_predictive_replicates(draws, 100, 50, 42)
        np.testing.assert_array_equal(np.random.rand(3), expected_numpy)
        torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0, atol=0)
        np.testing.assert_array_equal(first, posterior_predictive_replicates(draws, 100, 50, 42))
        np.testing.assert_array_equal(draws, original)
        self.assertFalse(np.array_equal(first, posterior_predictive_replicates(draws, 100, 50, 43)))
        with self.assertRaises(ValueError):
            posterior_predictive_replicates(draws, 100, 101, 42)

    def test_ppc_callback_preserves_all_existing_marginal_results(self):
        features, truth = simulate_fixed_n(4, 100, 2026)
        scaler = StandardScaler().fit(features)
        baseline = evaluate_cell(Posterior(), scaler, features, truth, 100, 100, 40, 42)
        def collect(index, draws):
            summarize_ppc(posterior_predictive_replicates(draws, 100, 20, index), features[index - 1])
        extended = evaluate_cell(Posterior(), scaler, features, truth, 100, 100, 40, 42, draw_callback=collect)
        pd.testing.assert_frame_equal(baseline, extended, check_exact=True)

    def test_summary_handles_extreme_observation(self):
        rows = summarize_ppc(np.tile(np.arange(5)[:, None], (1, 4)), [10, -1, 2, 2])
        self.assertEqual(rows[0]["upper_tail_fraction"], 0)
        self.assertEqual(rows[1]["upper_tail_fraction"], 1)
        self.assertEqual(rows[0]["observed_in_95_interval"], 0)
        self.assertEqual(rows[2]["observed_in_95_interval"], 1)

    def test_cell_resume_aggregate_and_replot_preserve_selected_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            cli = ["06_npe_diagnostics.py", "--run-all", "--device", "cpu",
                   "--k-values", "100", "--n-values", "100", "--n-test-datasets", "4",
                   "--n-posterior-draws", "40", "--n-ppc-replicates", "20",
                   "--inspection-dataset", "2"]
            with patch.object(sys, "argv", cli):
                args = diagnostics.parse_args()
            def train(features, ace, settings):
                return Posterior(), StandardScaler().fit(features)
            with patch.object(diagnostics, "train_transient_posterior", side_effect=train) as trainer:
                diagnostics.run_cell(100, 100, args, output)
                diagnostics.run_cell(100, 100, args, output)
                self.assertEqual(trainer.call_count, 1)
            saved = diagnostics.load_selected_samples(output, 100, 100, args)
            observed, truth = simulate_fixed_n(4, 100, diagnostics.seed_from(args.seed, 100, 20))
            np.testing.assert_array_equal(saved["observed"], observed[1])
            np.testing.assert_array_equal(saved["truth"], truth[1])
            ppc_path = output / "cells" / "K100_N100_ppc.csv"
            ppc_table = ppc_path.read_bytes()
            ppc_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing PPC"):
                diagnostics.aggregate(args, output)
            self.assertTrue((output / "cells").exists())
            self.assertFalse((output / "COMPLETE").exists())
            ppc_path.write_bytes(ppc_table)
            with patch.object(diagnostics, "save_all_figures"), patch.object(diagnostics, "save_inspection_figures") as plot:
                diagnostics.aggregate(args, output)
                self.assertFalse((output / "cells").exists())
                self.assertTrue((output / "COMPLETE").exists())
                self.assertEqual(len(pd.read_csv(output / "posterior_predictive_results.csv")), 16)
                args.inspection_dataset = 1
                args.n_ppc_replicates = 30
                diagnostics.replot(args, output)
                self.assertEqual(args.inspection_dataset, 2)
                self.assertEqual(args.n_ppc_replicates, 20)
                self.assertEqual(plot.call_count, 2)
            np.testing.assert_array_equal(saved["posterior"], diagnostics.load_selected_samples(output, 100, 100, args)["posterior"])


if __name__ == "__main__":
    unittest.main()
