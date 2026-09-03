from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from struct2circuit.sensitivity import (  # noqa: E402
    SensitivityConfig, conservative_sign_pvalue, holm_rejections,
    run_sensitivity, sign_median_lower_bound, stratified_bootstrap_median,
)


class SensitivityTests(unittest.TestCase):
    def test_sign_test_treats_ties_conservatively_and_matches_bound(self) -> None:
        values = np.asarray([-1.0] * 3 + [0.0] * 2 + [1.0] * 15)
        pvalue = conservative_sign_pvalue(values)
        bound = sign_median_lower_bound(values, 0.05)
        self.assertEqual(pvalue <= 0.05, bound > 0)
        self.assertGreater(conservative_sign_pvalue(np.zeros(20)), 0.05)

    def test_stratified_bootstrap_is_deterministic_and_preserves_shape(self) -> None:
        values = np.arange(16, dtype=float)
        cells = np.repeat(np.arange(4), 4)
        first = stratified_bootstrap_median(values, cells, samples=25, seed=7)
        second = stratified_bootstrap_median(values, cells, samples=25, seed=7)
        self.assertEqual(first.shape, (25,))
        self.assertTrue(np.array_equal(first, second))

    def test_holm_stops_after_first_non_rejection(self) -> None:
        self.assertEqual(holm_rejections(np.asarray([0.01, 0.04, 0.9]), 0.05).tolist(), [True, False, False])

    def test_small_run_is_deterministic_and_synthetic(self) -> None:
        config = SensitivityConfig(
            seed=3, bootstrap_samples=19, min_repetitions=2,
            max_repetitions=2, batch_size=2, target_mcse=1,
            runtime_cap_seconds=30,
        )
        kwargs = {"family_totals": (32,), "effects": (0.0, 0.01), "scenarios": ("gaussian", "optimizer_failure")}
        first = run_sensitivity(config, **kwargs)
        second = run_sensitivity(config, **kwargs)
        for result in (first, second):
            result.pop("runtime_seconds")
        self.assertEqual(first, second)
        self.assertEqual(first["completed_grid_points"], first["planned_grid_points"])
        self.assertEqual(first["status"], "synthetic_sensitivity_only_not_benchmark_freeze")

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            run_sensitivity(SensitivityConfig(min_repetitions=3, max_repetitions=2))


if __name__ == "__main__":
    unittest.main()
