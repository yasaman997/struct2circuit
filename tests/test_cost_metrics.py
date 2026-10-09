from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from struct2circuit.mixers import ring_mixer
from struct2circuit.optimize import optimize_p1
from struct2circuit.problems import CardinalityQUBO
from struct2circuit.simulator import FeasibleSubspaceQAOA


class CostMetricTests(unittest.TestCase):
    def simulator(self, c, scale=1.0, shift=0.0):
        c = np.asarray(c, dtype=float)
        return FeasibleSubspaceQAOA(
            CardinalityQUBO(np.zeros((len(c), len(c))), scale * c + shift / 2, 2),
            ring_mixer(len(c)),
        )

    def test_unique_and_multiple_optima_are_invariant_to_units(self):
        for coefficients, optimum_count in [([0, 1, 3, 8], 1), ([0, 1, 1, 4], 2)]:
            reference = self.simulator(coefficients)
            expected = reference.optimal_mask
            self.assertEqual(int(expected.sum()), optimum_count)
            for scale in (np.nextafter(0.0, 1.0), 1e-310, 1e-200, 1e-16, 1e-12,
                          1e-4, 1.0, 7.0, 100.0, 1e12, 1e150):
                with self.subTest(coefficients=coefficients, scale=scale):
                    sim = self.simulator(coefficients, scale=scale)
                    self.assertEqual(sim.cost_status, "nonconstant")
                    np.testing.assert_array_equal(sim.optimal_mask, expected)
                    np.testing.assert_allclose(sim.normalized_costs, reference.normalized_costs,
                                               rtol=0, atol=5e-16)
                    result = sim.evaluate_dimensionless([1.17], [0.4])
                    original = reference.evaluate_dimensionless([1.17], [0.4])
                    self.assertAlmostEqual(result.normalized_gap, original.normalized_gap, places=14)
                    self.assertAlmostEqual(result.probability_optimum, original.probability_optimum,
                                           places=14)

    def test_additive_constant_preserves_normalized_metrics(self):
        reference = self.simulator([0, 1, 3, 8])
        expected = reference.evaluate_dimensionless([2.7], [0.9])
        for shift in (-1e8, -4.0, 4.0, 1e4, 1e8):
            with self.subTest(shift=shift):
                sim = self.simulator([0, 1, 3, 8], shift=shift)
                np.testing.assert_array_equal(sim.optimal_mask, reference.optimal_mask)
                result = sim.evaluate_dimensionless([2.7], [0.9])
                self.assertAlmostEqual(result.normalized_gap, expected.normalized_gap, places=14)
                self.assertAlmostEqual(result.probability_optimum, expected.probability_optimum,
                                       places=14)
                self.assertAlmostEqual(result.expectation - shift, expected.expectation, places=6)

    def test_dimensionless_state_is_physical_state_up_to_global_phase(self):
        rng = np.random.default_rng(9019)
        q = rng.normal(size=(5, 5))
        p = CardinalityQUBO((q + q.T) / 2, rng.normal(size=5), 2)
        sim = FeasibleSubspaceQAOA(p, ring_mixer(5))
        us = np.array([0.6])
        betas = np.array([0.4])
        raw = sim.state(us / sim.feasible_span, betas)
        normalized = sim.state_dimensionless(us, betas)
        phase = np.exp(1j * np.sum(us) * sim.cost_min / sim.feasible_span)
        np.testing.assert_allclose(normalized, phase * raw, rtol=0, atol=2e-14)
        self.assertAlmostEqual(sim.evaluate(us / sim.feasible_span, betas).normalized_gap,
                               sim.evaluate_dimensionless(us, betas).normalized_gap, places=14)

    def test_constant_and_unresolved_gaps_serialize_as_null(self):
        for coefficients in ([0, 0, 0, 0], [3, 3, 3, 3]):
            sim = self.simulator(coefficients)
            self.assertEqual(sim.cost_status, "constant_or_unresolved")
            self.assertIsNone(sim.normalized_costs)
            for result in (sim.evaluate([0.5], [0.3]),
                           sim.evaluate_dimensionless([0.0], [0.0])):
                self.assertIsNone(result.normalized_gap)
                self.assertIsNone(result.probability_optimum)
                serialized = json.loads(json.dumps(asdict(result), allow_nan=False))
                self.assertIsNone(serialized["normalized_gap"])
                self.assertEqual(serialized["cost_status"], "constant_or_unresolved")

    def test_variation_lost_to_input_rounding_is_not_reported_as_success(self):
        # These distinctions are below float64 resolution at this offset. The
        # simulator must not infer true constancy or claim an optimality gap of 0.
        sim = self.simulator([0, 1, 3, 8], shift=1e20)
        self.assertEqual(sim.cost_status, "constant_or_unresolved")
        result = sim.evaluate_dimensionless([0.0], [0.0])
        self.assertIsNone(result.normalized_gap)
        self.assertIsNone(result.probability_optimum)

    def test_exact_constant_polynomial_does_not_normalize_evaluation_noise(self):
        v = 1 + np.array([-4., -4., 0., 1.]) * 2.0**-51
        q = (v[:, None] + v[None, :]) / 2
        c = -2 * v
        for scale, shift in ((2.0**-600, 0.0), (1.0, 0.0), (1.0, 0.25),
                             (2.0**500, 0.0)):
            with self.subTest(scale=scale, shift=shift):
                problem = CardinalityQUBO(scale * q, scale * c + shift / 2, 2)
                sim = FeasibleSubspaceQAOA(problem, ring_mixer(4))
                # Independent exact arithmetic verifies the represented inputs,
                # rather than assuming a symbolic identity survived rounding.
                for state in sim.basis:
                    occupied = np.flatnonzero(state)
                    exact = sum(Fraction.from_float(float(problem.Q[i, j]))
                                for i in occupied for j in occupied)
                    exact += sum(Fraction.from_float(float(problem.c[i]))
                                 for i in occupied)
                    self.assertEqual(exact, Fraction.from_float(shift))
                np.testing.assert_array_equal(sim.costs, problem.costs(sim.basis))
                if scale == 1.0 and shift == 0.0:
                    self.assertGreater(float(np.ptp(sim.costs)), 0.0)
                self.assertEqual(sim.feasible_span, 0.0)
                self.assertEqual(sim.cost_status, "constant_or_unresolved")
                self.assertIsNone(sim.normalized_costs)
                result = optimize_p1(sim, grid_size=5)
                self.assertEqual(result.selected_source, "not_run")
                self.assertEqual(result.objective_evaluations, 0)
                self.assertIsNone(result.result.normalized_gap)
                self.assertIsNone(result.result.probability_optimum)

    def test_small_resolved_variation_survives_cancellation(self):
        v = 1 + np.array([-4., -4., 0., 1.]) * 2.0**-51
        q = (v[:, None] + v[None, :]) / 2
        c = -2 * v
        c[-1] += 2.0**-49
        sim = FeasibleSubspaceQAOA(CardinalityQUBO(q, c, 2), ring_mixer(4))
        self.assertEqual(sim.cost_status, "nonconstant")
        self.assertEqual(sim.feasible_span, 2.0**-49)
        np.testing.assert_array_equal(sim.normalized_costs, sim.basis[:, -1])

    def test_representable_variation_at_large_offset_remains_resolved(self):
        for shift in (0.0, float(2**52)):
            with self.subTest(shift=shift):
                sim = FeasibleSubspaceQAOA(CardinalityQUBO(
                    np.zeros((4, 4)), shift + np.array([0., 1., 3., 6.]), 1),
                    ring_mixer(4))
                self.assertEqual(sim.cost_status, "nonconstant")
                self.assertEqual(sim.feasible_span, 6.0)
                np.testing.assert_array_equal(sim.normalized_costs,
                                              np.array([0., 1., 3., 6.]) / 6)

    def test_variation_lost_in_final_cost_rounding_is_withheld(self):
        sim = FeasibleSubspaceQAOA(CardinalityQUBO(
            1e16 * np.eye(4), np.array([0., 0.125, 0.25, 0.5]), 2), ring_mixer(4))
        self.assertEqual(sim.cost_status, "constant_or_unresolved")
        self.assertIsNone(sim.evaluate_dimensionless([0.0], [0.0]).normalized_gap)

    def test_resolved_near_minimum_is_excluded_across_objective_scales(self):
        for scale in (1e-16, 1.0, 1e16):
            sim = self.simulator([0, 1, 1 + 1e-7, 4], scale=scale)
            self.assertEqual(int(sim.optimal_mask.sum()), 1)

    def test_nonfinite_costs_are_explicitly_rejected(self):
        with np.errstate(invalid="ignore"):
            with self.assertRaisesRegex(ValueError, "finite floating-point range"):
                self.simulator([0, 1, np.inf, 4])

    def test_unrepresentable_span_is_explicitly_rejected(self):
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaisesRegex(ValueError, "span exceeds"):
                FeasibleSubspaceQAOA(
                    CardinalityQUBO(np.zeros((2, 2)), np.array([-1e308, 1e308]), 1),
                    ring_mixer(2),
                )


if __name__ == "__main__":
    unittest.main()
