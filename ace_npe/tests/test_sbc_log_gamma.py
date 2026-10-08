"""Numerical and statistical checks for the saved-rank log-gamma metric."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbc_log_gamma import (
    gamma_null_distribution, gamma_null_threshold, log_gamma_discrepancy,
    summarize_log_gamma,
)


def bayesflow_reference(ranks, n_draws):
    """Literal equation/source reference, suitable for non-extreme examples."""
    m = len(ranks)
    counts = np.array([sum(ranks < j) for j in range(1, n_draws + 2)])
    grid = np.arange(1, n_draws + 2) / (n_draws + 1)
    return 2 * np.min(np.minimum(binom.cdf(counts, m, grid),
                                 1 - binom.cdf(counts - 1, m, grid)))


class LogGammaChecks(unittest.TestCase):
    def test_matches_bayesflow_formula_with_zero_and_L_ranks(self):
        rng = np.random.default_rng(2026)
        for n_draws in (1, 20, 100):
            ranks = np.concatenate(([0, n_draws], rng.integers(0, n_draws + 1, 78)))
            expected = bayesflow_reference(ranks, n_draws)
            self.assertAlmostEqual(np.exp(log_gamma_discrepancy(ranks, n_draws)), expected, places=13)

    def test_null_matches_uniform_ecdf_reference_and_leaves_global_rng_alone(self):
        seed, m, l = 42, 60, 20
        rng = np.random.default_rng(seed)
        expected = []
        for _ in range(20):
            # Discrete ranks derived from the same uniforms independently
            # reproduce the source's continuous-uniform ECDF grid.
            ranks = np.floor((l + 1) * rng.random(m)).astype(int)
            expected.append(bayesflow_reference(ranks, l))
        np.random.seed(123)
        expected_random = np.random.random(5)
        np.random.seed(123)
        actual = gamma_null_distribution(m, l, 20, seed)
        np.testing.assert_array_equal(np.random.random(5), expected_random)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(actual, gamma_null_distribution(m, l, 20, seed))

    def test_five_percent_threshold_has_approximately_five_percent_null_rejections(self):
        threshold = gamma_null_threshold(100, 40, 1000, 0.05, 42)
        independent_null = gamma_null_distribution(100, 40, 3000, 123)
        rate = np.mean(independent_null < threshold)
        self.assertGreater(rate, 0.025)
        self.assertLess(rate, 0.085)

    def test_extreme_departure_keeps_finite_log_score_when_gamma_underflows(self):
        ranks = np.zeros(1000, dtype=int)
        expected = np.log(2) - 1000 * np.log(2001)
        self.assertAlmostEqual(log_gamma_discrepancy(ranks, 2000), expected, places=9)
        summary = summarize_log_gamma(ranks, 2000)
        self.assertEqual(summary["gamma"], 0)
        self.assertTrue(np.isfinite(summary["log_gamma"]))
        self.assertLess(summary["log_gamma"], -1000)
        self.assertTrue(summary["rejects_uniformity"])

    def test_rejects_invalid_ranks_and_quantile(self):
        for ranks in ([0], [0, 21], [-1, 0], [1.5, 3], [np.nan, 1], [[1, 2]]):
            with self.subTest(ranks=ranks), self.assertRaises(ValueError):
                log_gamma_discrepancy(ranks, 20)
        with self.assertRaises(ValueError):
            gamma_null_threshold(100, 20, 100, 0, 42)


if __name__ == "__main__":
    unittest.main()
