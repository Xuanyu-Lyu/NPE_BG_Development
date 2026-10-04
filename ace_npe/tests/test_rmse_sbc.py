"""Regression checks for the added RMSE SBC and preservation of marginal SBC.

Run from the repository root:
    python -m unittest discover -s ace_npe/tests -v
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
from rmse_sbc import covariance_prediction_rmse, summarize_rmse_sbc
from training_budget_utils import evaluate_cell, simulate_fixed_n

spec = importlib.util.spec_from_file_location(
    "npe_diagnostics", Path(__file__).resolve().parents[1] / "06_npe_diagnostics.py"
)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class RMSESBCChecks(unittest.TestCase):
    def setUp(self):
        self.observed = np.array([0.98, 0.72, 1.02, 0.47])
        self.truth = np.array([0.4, 0.3, 0.3])
        self.draws = np.array([
            [0.50, 0.22, 0.28], [0.44, 0.27, 0.29],
            [0.55, 0.15, 0.30], [0.34, 0.39, 0.27],
        ])

    def test_worked_example_and_rank(self):
        # Independently calculated squared errors from the discussion example.
        expected = np.sqrt(np.array([0.0008, 0.0013, 0.003225, 0.009]) / 4)
        np.testing.assert_allclose(
            covariance_prediction_rmse(self.draws, self.observed), expected
        )
        summary = summarize_rmse_sbc(
            self.draws, self.truth, self.observed, np.random.default_rng(42)
        )
        self.assertAlmostEqual(summary["true_rmse"], np.sqrt(0.0021 / 4))
        self.assertEqual(summary["true_rank"], 2)
        self.assertEqual(summary["rank_ties"], 0)
        self.assertEqual(summary["true_rank_percentile"], 0.5)

    def test_rank_endpoints_are_inclusive(self):
        exact_observation = np.array([1.0, 0.7, 1.0, 0.5])
        low = summarize_rmse_sbc(
            self.draws, self.truth, exact_observation, np.random.default_rng(42)
        )
        high = summarize_rmse_sbc(
            self.draws[:2], [0.1, 0.1, 0.8], self.observed, np.random.default_rng(42)
        )
        self.assertEqual(low["true_rank"], 0)
        self.assertEqual(high["true_rank"], 2)

    def test_exact_ties_are_randomized_reproducibly(self):
        draws = np.tile(self.truth, (4, 1))
        def ranks(seed):
            rng = np.random.default_rng(seed)
            return [summarize_rmse_sbc(draws, self.truth, self.observed, rng)["true_rank"]
                    for _ in range(100)]
        self.assertEqual(ranks(42), ranks(42))
        self.assertEqual(set(ranks(42)), set(range(5)))
        summary = summarize_rmse_sbc(draws, self.truth, self.observed, np.random.default_rng(42))
        self.assertEqual(summary["rank_strict"], 0)
        self.assertEqual(summary["rank_ties"], 4)

    def test_exchangeable_draws_have_uniform_ranks(self):
        # With a fixed observation and iid theta draws, truth and posterior
        # quantities are exchangeable. This isolates the rank construction.
        rng = np.random.default_rng(2026)
        ranks = []
        for _ in range(2000):
            theta = rng.dirichlet(np.ones(3), size=21)
            ranks.append(summarize_rmse_sbc(theta[1:], theta[0], self.observed, rng)["true_rank"])
        proportions = np.bincount(ranks, minlength=21) / 2000
        self.assertLess(float(np.max(np.abs(proportions - 1 / 21))), 0.025)

    def test_rejects_nonfinite_or_non_simplex_inputs(self):
        for theta, observed in (
            ([0.4, 0.3, 0.4], self.observed),
            ([-0.1, 0.4, 0.7], self.observed),
            (self.truth, [1, np.nan, 1, 0.5]),
            (self.truth, [1, 0.7]),
        ):
            with self.subTest(theta=theta, observed=observed), self.assertRaises(ValueError):
                covariance_prediction_rmse(theta, observed)

    def test_callback_preserves_marginal_results_and_sampling(self):
        class Posterior:
            def __init__(self):
                self.calls = 0

            def sample(self, shape, **kwargs):
                self.calls += 1
                return torch.softmax(torch.randn(shape[0], 3), dim=-1)

        features, truths = simulate_fixed_n(5, 100, 2026)
        scaler = StandardScaler().fit(features)
        plain_posterior, extended_posterior = Posterior(), Posterior()
        baseline = evaluate_cell(plain_posterior, scaler, features, truths, 100000, 100, 40, 42)
        rmse_rows = []
        rng = np.random.default_rng(123)

        def collect(index, draws):
            rmse_rows.append(summarize_rmse_sbc(draws, truths[index - 1], features[index - 1], rng))

        extended = evaluate_cell(
            extended_posterior, scaler, features, truths, 100000, 100, 40, 42,
            draw_callback=collect,
        )
        pd.testing.assert_frame_equal(baseline, extended, check_exact=True)
        self.assertEqual(plain_posterior.calls, 5)
        self.assertEqual(extended_posterior.calls, 5)
        self.assertEqual(len(rmse_rows), 5)

    def test_missing_rmse_cannot_complete_or_delete_cells(self):
        args = argparse.Namespace(k_values=(100000,), n_values=(100,), n_test_datasets=2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            cells = output / "cells"
            cells.mkdir()
            (cells / "K100000_N100.csv").write_text("old marginal results\n")
            (cells / "K100000_N100.json").write_text("{}\n")
            with self.assertRaisesRegex(FileNotFoundError, "joint posterior draws were discarded"):
                diagnostics.aggregate(args, output)
            self.assertTrue(cells.exists())
            self.assertFalse((output / "COMPLETE").exists())

    def test_resume_rejects_changed_sampling_or_training_settings(self):
        with patch.object(sys, "argv", ["06_npe_diagnostics.py", "--cell-index", "1", "--device", "cpu"]):
            args = diagnostics.parse_args()
        expected = diagnostics.expected_cell_metadata(100000, 50, args)
        metadata = json.loads(json.dumps(expected))
        diagnostics.validate_cell_metadata(metadata, expected, Path("cell.json"))
        metadata["L"] = 1000
        with self.assertRaisesRegex(ValueError, "incompatible"):
            diagnostics.validate_cell_metadata(metadata, expected, Path("cell.json"))


if __name__ == "__main__":
    unittest.main()
