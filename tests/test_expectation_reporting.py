from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from struct2circuit.mixers import ring_mixer
from struct2circuit.problems import CardinalityQUBO
from struct2circuit.simulator import FeasibleSubspaceQAOA


def exact_costs(problem):
    return [sum((Fraction.from_float(float(problem.Q[i, j]))
                 for i in occupied for j in occupied), Fraction(0))
            + sum((Fraction.from_float(float(problem.c[i])) for i in occupied), Fraction(0))
            for occupied in combinations(range(problem.n), problem.k)]


class ExpectationReportingTests(unittest.TestCase):
    def simulator(self, q, c, k):
        return FeasibleSubspaceQAOA(CardinalityQUBO(q, c, k), ring_mixer(len(c)))

    def cancellation_problem(self, residual=0., scale=1.):
        v = 1 + np.array([-4., -4., 0., 1.]) * 2.0**-51
        q = (v[:, None] + v[None, :]) / 2
        c = -2 * v
        c[-1] += residual
        return self.simulator(scale * q, scale * c, 2)

    def assert_normalized_oracle(self, sim, state):
        costs = np.array([float(cost) for cost in exact_costs(sim.problem)])
        probabilities = np.abs(state)**2
        result = sim._evaluate_state(state)
        if min(costs) == max(costs):
            self.assertIsNone(result.normalized_gap)
            self.assertIsNone(result.probability_optimum)
        else:
            normalized = (costs - min(costs)) / (max(costs) - min(costs))
            self.assertEqual(result.normalized_gap, float(probabilities @ normalized))
            self.assertEqual(result.probability_optimum,
                             float(probabilities[costs == min(costs)].sum()))
        return result

    def test_moderate_costs_match_raw_and_compensated_representations(self):
        q = np.array([[2., 1., -1., 0.], [1., 0., 2., 0.],
                      [-1., 2., 1., 1.], [0., 0., 1., 3.]])
        sim = self.simulator(q, [1., -2., 0., 2.], 2)
        weights = np.arange(1, 7, dtype=float)
        state = np.sqrt(weights / weights.sum()).astype(complex)
        values = np.array([float(cost) for cost in exact_costs(sim.problem)])
        result = self.assert_normalized_oracle(sim, state)
        self.assertEqual(result.expectation, float((np.abs(state)**2) @ values))
        self.assertAlmostEqual((result.expectation - sim.cost_min) / sim.feasible_span,
                               result.normalized_gap)
        self.assertEqual(set(asdict(result)), {"expectation", "normalized_gap",
                         "probability_optimum", "state_norm", "feasibility_probability",
                         "cost_status"})

    def test_exact_cancellation_retains_raw_noise_and_withholds_normalized_metrics(self):
        sim = self.cancellation_problem()
        self.assertTrue(all(cost == 0 for cost in exact_costs(sim.problem)))
        state = np.eye(6, dtype=complex)[-1]
        result = self.assert_normalized_oracle(sim, state)
        self.assertEqual(result.expectation, -2.0**-50)
        self.assertEqual(result.cost_status, "constant_or_unresolved")

    def test_tiny_resolved_span_does_not_use_raw_expectation_to_compute_gap(self):
        sim = self.cancellation_problem(2.0**-49)
        state = np.eye(6, dtype=complex)[-1]
        result = self.assert_normalized_oracle(sim, state)
        self.assertEqual(sim.feasible_span, 2.0**-49)
        self.assertEqual(result.expectation, 2.0**-50)
        self.assertEqual(result.normalized_gap, 1.)
        self.assertEqual((result.expectation - sim.cost_min) / sim.feasible_span, .5)
        self.assertEqual(result.probability_optimum, 0.)

    def test_large_offsets_preserve_the_declared_final_rounding_ties(self):
        for offset, c in ((2.0**52, [0., 1., 3.]), (1e16, [0., .125, 4.])):
            with self.subTest(offset=offset):
                sim = self.simulator(offset * np.eye(3), c, 1)
                state = np.sqrt([.25, .5, .25]).astype(complex)
                result = self.assert_normalized_oracle(sim, state)
                self.assertEqual(result.expectation, float((np.abs(state)**2) @ sim.costs))
                if offset == 1e16:
                    self.assertAlmostEqual(result.probability_optimum, .75, places=15)

    def test_scaling_preserves_normalized_metrics_without_reinterpreting_raw_values(self):
        reference = self.cancellation_problem(2.0**-49)
        state = np.eye(6, dtype=complex)[-1]
        expected = self.assert_normalized_oracle(reference, state)
        for exponent in (-500, -40, 40, 500):
            with self.subTest(exponent=exponent):
                scale = 2.0**exponent
                sim = self.cancellation_problem(2.0**-49, scale)
                result = self.assert_normalized_oracle(sim, state)
                self.assertEqual(result.expectation, scale * expected.expectation)
                self.assertEqual(result.normalized_gap, expected.normalized_gap)
                self.assertEqual(result.probability_optimum, expected.probability_optimum)

    def test_constant_and_unresolved_reporting_remains_finite_and_serializable(self):
        for sim in (self.simulator(np.zeros((4, 4)), [3., 3., 3., 3.], 2),
                    self.simulator(1e16 * np.eye(4), [0., .125, .25, .5], 2)):
            with self.subTest(cost_min=sim.cost_min):
                result = self.assert_normalized_oracle(sim, sim.uniform_state)
                self.assertTrue(np.isfinite(result.expectation))
                serialized = json.loads(json.dumps(asdict(result), allow_nan=False))
                self.assertIsNone(serialized["normalized_gap"])
                self.assertIsNone(serialized["probability_optimum"])

    def test_one_ulp_cost_differences_use_the_compensated_normalized_oracle(self):
        spacing = np.spacing(1.)
        sim = self.simulator(np.eye(3), [0., spacing, 4 * spacing], 1)
        state = np.eye(3, dtype=complex)[1]
        result = self.assert_normalized_oracle(sim, state)
        self.assertEqual(result.expectation, 1. + spacing)
        self.assertEqual(result.normalized_gap, .25)
        self.assertEqual(result.probability_optimum, 0.)

    def test_legacy_raw_cost_phases_survive_exact_cancellation(self):
        sim = self.cancellation_problem()
        gamma = 2.0**50
        expected = sim.uniform_state.copy()
        expected[-1] *= np.exp(1j)
        raw = sim.state([gamma], [0.])
        np.testing.assert_allclose(raw, expected, rtol=0, atol=2e-15)
        normalized = sim.state_dimensionless([1.], [0.])
        np.testing.assert_allclose(normalized, sim.uniform_state, rtol=0, atol=2e-15)
        self.assertGreater(float(np.linalg.norm(raw - normalized)), .1)
        result = sim.evaluate([gamma], [0.])
        self.assertEqual(result.expectation, float((np.abs(raw)**2) @ sim.costs))
        self.assertIsNone(result.normalized_gap)


if __name__ == "__main__":
    unittest.main()
