from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from struct2circuit.sensitivity import (  # noqa: E402
    DISPERSION_MULTIPLIERS, SCENARIOS, SensitivityConfig,
    conservative_mc_precision, conservative_sign_pvalue, generate_synthetic_comparison,
    generate_synthetic_repetition,
    holm_rejections, joint_lower_bound_coverage, null_control_truth,
    run_sensitivity, sign_median_lower_bound, stratified_bootstrap_median,
)


class SensitivityTests(unittest.TestCase):
    def test_sign_test_treats_ties_conservatively_and_matches_bound(self) -> None:
        values = np.asarray([-1.0] * 3 + [0.0] * 2 + [1.0] * 15)
        self.assertEqual(
            conservative_sign_pvalue(values) <= 0.05,
            sign_median_lower_bound(values, 0.05) > 0,
        )
        self.assertGreater(conservative_sign_pvalue(np.zeros(20)), 0.05)

    def test_stratified_bootstrap_is_deterministic(self) -> None:
        values = np.arange(16, dtype=float)
        cells = np.repeat(np.arange(4), 4)
        first = stratified_bootstrap_median(values, cells, samples=25, seed=7)
        second = stratified_bootstrap_median(values, cells, samples=25, seed=7)
        self.assertEqual(first.shape, (25,))
        self.assertTrue(np.array_equal(first, second))

    def test_holm_stops_after_first_non_rejection(self) -> None:
        decisions = holm_rejections(np.asarray([0.01, 0.04, 0.9]), 0.05)
        self.assertEqual(decisions.tolist(), [True, False, False])

    def test_null_population_is_independent_of_structured_effects(self) -> None:
        self.assertEqual(null_control_truth((0.0, 0.0, 0.0)), (0.0, 0.0))
        self.assertEqual(null_control_truth((0.015, -0.2, 0.8)), (0.0, 0.0))
        config = SensitivityConfig()
        structured, _ = generate_synthetic_comparison(
            seed=11, total=32, effect=0.015, scenario="gaussian",
            dispersion="medium", config=config,
        )
        null, _ = generate_synthetic_comparison(
            seed=12, total=32, effect=0.0, scenario="gaussian",
            dispersion="medium", config=config,
        )
        self.assertFalse(np.array_equal(structured.ring, null.ring))
        self.assertEqual((null.ring_truth, null.expected_random_truth), (0.0, 0.0))
        common = dict(
            seed=77, total=32, scenario="gaussian", dispersion="medium",
            config=config,
        )
        _, first_null, _ = generate_synthetic_repetition(
            structured_effects=(0.0, 0.0, 0.0), **common,
        )
        _, second_null, _ = generate_synthetic_repetition(
            structured_effects=(0.015, -0.4, 0.8), **common,
        )
        self.assertTrue(np.array_equal(first_null.ring, second_null.ring))
        self.assertTrue(np.array_equal(first_null.expected_random, second_null.expected_random))

    def test_null_truth_is_zero_for_every_scenario_and_dispersion(self) -> None:
        config = SensitivityConfig(failure_rate=1.0)
        for scenario in SCENARIOS:
            for dispersion in DISPERSION_MULTIPLIERS:
                with self.subTest(scenario=scenario, dispersion=dispersion):
                    _, null, _ = generate_synthetic_repetition(
                        seed=83, total=32,
                        structured_effects=(0.015, 0.010, 0.005),
                        scenario=scenario, dispersion=dispersion, config=config,
                    )
                    self.assertEqual(
                        (null.ring_truth, null.expected_random_truth),
                        null_control_truth(),
                    )
                    if scenario == "structure_failure":
                        self.assertFalse(np.all(null.ring == config.structure_failure_value))

    def test_random_baseline_uncertainty_affects_only_random_comparison(self) -> None:
        zero = SensitivityConfig(random_baseline_sd=0.0)
        positive = SensitivityConfig(random_baseline_sd=0.02)
        kwargs = dict(seed=19, total=32, effect=0.01, scenario="gaussian", dispersion="medium")
        no_error, _ = generate_synthetic_comparison(config=zero, **kwargs)
        with_error, _ = generate_synthetic_comparison(config=positive, **kwargs)
        self.assertTrue(np.array_equal(no_error.ring, no_error.expected_random))
        self.assertTrue(np.array_equal(no_error.ring, with_error.ring))
        self.assertFalse(np.array_equal(with_error.ring, with_error.expected_random))
        self.assertEqual(with_error.ring_truth, with_error.expected_random_truth)

    def test_joint_coverage_is_not_average_marginal_coverage(self) -> None:
        lower = np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, 0.1])
        truths = np.zeros(6)
        self.assertAlmostEqual(np.mean(lower <= truths), 5 / 6)
        self.assertFalse(joint_lower_bound_coverage(lower, truths))
        family_lower = np.asarray([-1.0, -1.0, 0.1])
        self.assertFalse(joint_lower_bound_coverage(family_lower, np.zeros(3)))

    def test_failure_conventions_retain_neutral_and_adverse_values(self) -> None:
        config = SensitivityConfig(failure_rate=1.0)
        common = dict(seed=2, total=32, effect=0.015, dispersion="low", config=config)
        neutral, _ = generate_synthetic_comparison(scenario="optimizer_failure", **common)
        adverse, _ = generate_synthetic_comparison(scenario="structure_failure", **common)
        self.assertTrue(np.all(neutral.ring == config.neutral_failure_value))
        self.assertTrue(np.all(neutral.expected_random == config.neutral_failure_value))
        self.assertTrue(np.all(adverse.ring == config.structure_failure_value))
        self.assertTrue(np.all(adverse.expected_random == config.structure_failure_value))
        self.assertLess(adverse.ring_truth, neutral.ring_truth)

    def test_small_run_is_deterministic_and_has_dispersion_metadata(self) -> None:
        config = SensitivityConfig(
            seed=3, bootstrap_samples=19, min_repetitions=2,
            max_repetitions=2, batch_size=2, target_mcse=1,
            runtime_cap_seconds=30,
        )
        kwargs = {
            "family_totals": (32,), "effects": (0.0, 0.01),
            "scenarios": ("gaussian", "structure_failure"),
            "dispersions": ("low", "high"),
        }
        first = run_sensitivity(config, **kwargs)
        second = run_sensitivity(config, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["completed_grid_points"], first["planned_grid_points"])
        self.assertEqual({row["dispersion"] for row in first["rows"]}, {"low", "high"})
        self.assertEqual(
            first["design"]["null_control_population_targets"],
            {"ring": 0.0, "expected_random": 0.0},
        )

    def test_endpoint_rates_do_not_claim_zero_precision_or_stop_early(self) -> None:
        config = SensitivityConfig(
            seed=5, bootstrap_samples=19, min_repetitions=2,
            max_repetitions=5, batch_size=1, target_mcse=0.10,
            runtime_cap_seconds=30, failure_rate=1.0,
        )
        result = run_sensitivity(
            config, family_totals=(32,), effects=(0.0,),
            scenarios=("structure_failure",), dispersions=("low",),
        )
        self.assertTrue(all(row["repetitions"] == 5 for row in result["rows"]))
        self.assertTrue(all(row["stopping_reason"] == "max_repetitions" for row in result["rows"]))
        self.assertTrue(all(row["achieved_mc_precision"] > config.target_mcse for row in result["rows"]))
        self.assertTrue(all(row["success_mcse"] > 0 for row in result["rows"]))
        self.assertTrue(all(row["fwer_mcse"] > 0 for row in result["rows"] if row["fwer_mcse"] is not None))
        self.assertAlmostEqual(conservative_mc_precision(5), 0.5 / np.sqrt(5))

    def test_runtime_cap_before_first_repetition_returns_valid_partial_result(self) -> None:
        config = SensitivityConfig(runtime_cap_seconds=0.5)
        with patch("struct2circuit.sensitivity.time.monotonic", side_effect=[0.0, 1.0]):
            result = run_sensitivity(
                config, family_totals=(32,), effects=(0.0,),
                scenarios=("gaussian",), dispersions=("low",),
            )
        self.assertEqual(result["status"], "partial_runtime_cap_reached")
        self.assertEqual(result["completed_grid_points"], 0)
        self.assertEqual(result["rows"], [])

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            run_sensitivity(SensitivityConfig(min_repetitions=3, max_repetitions=2))


if __name__ == "__main__":
    unittest.main()
