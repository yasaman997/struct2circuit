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


def exact_feasible_costs(problem):
    """Enumerate actual costs from represented coefficients, without simulator sums."""
    values = {}
    for occupied in combinations(range(problem.n), problem.k):
        state = tuple(int(index in occupied) for index in range(problem.n))
        quadratic = sum(
            (Fraction.from_float(float(problem.Q[i, j]))
             for i in occupied for j in occupied), Fraction(0)
        )
        linear = sum(
            (Fraction.from_float(float(problem.c[i])) for i in occupied), Fraction(0)
        )
        values[state] = quadratic + linear
    return values


class OptimumProbabilityTests(unittest.TestCase):
    def simulator(self, coefficients, k=1, *, q=None, initialization="uniform"):
        coefficients = np.asarray(coefficients, dtype=float)
        if q is None:
            q = np.zeros((len(coefficients), len(coefficients)))
        return FeasibleSubspaceQAOA(
            CardinalityQUBO(q, coefficients, k), ring_mixer(len(coefficients)),
            initialization=initialization,
        )

    def assert_exact_optimal_mask(self, sim):
        costs = exact_feasible_costs(sim.problem)
        optimum = min(costs.values())
        expected = np.array([costs[tuple(state)] == optimum for state in sim.basis])
        np.testing.assert_array_equal(sim.optimal_mask, expected)
        return expected

    def assert_probability_from_oracle(self, sim, state):
        expected_mask = self.assert_exact_optimal_mask(sim)
        probabilities = np.abs(state) ** 2
        result = sim._evaluate_state(state)
        self.assertAlmostEqual(
            result.probability_optimum, float(probabilities[expected_mask].sum()), places=14
        )
        return result

    def test_unique_optimum_matches_exact_quadratic_cost_enumeration(self):
        q = np.array([[2, 1, -1, 0], [1, 0, 2, 0],
                      [-1, 2, 1, 1], [0, 0, 1, 3]], dtype=float)
        sim = self.simulator([1, -2, 0, 2], k=2, q=q)
        expected = self.assert_exact_optimal_mask(sim)
        self.assertEqual(int(expected.sum()), 1)
        np.testing.assert_array_equal(sim.basis[expected][0], [1, 0, 1, 0])
        weights = np.arange(1, sim.feasible_dimension + 1, dtype=float)
        state = np.sqrt(weights / weights.sum()).astype(complex)
        self.assert_probability_from_oracle(sim, state)

    def test_multiple_exact_optima_sum_all_optimal_probability_mass(self):
        sim = self.simulator([0, 1, 1, 4], k=2)
        expected = self.assert_exact_optimal_mask(sim)
        self.assertEqual(int(expected.sum()), 2)
        weights = np.arange(1, sim.feasible_dimension + 1, dtype=float)
        probabilities = weights / weights.sum()
        state = np.sqrt(probabilities).astype(complex)
        result = self.assert_probability_from_oracle(sim, state)
        self.assertAlmostEqual(result.probability_optimum, 3 / 21, places=15)

    def test_strictly_suboptimal_state_below_old_tolerance_is_excluded(self):
        sim = self.simulator([0, 2.0**-40, 1])
        exact_costs = exact_feasible_costs(sim.problem)
        self.assertGreater(exact_costs[(0, 1, 0)], exact_costs[(1, 0, 0)])
        self.assertGreater(sim.normalized_costs[1], 0.0)
        self.assertLess(sim.normalized_costs[1], 1e-10)
        old_mask = np.isclose(sim.normalized_costs, 0.0, rtol=0.0, atol=1e-10)
        np.testing.assert_array_equal(old_mask, [True, True, False])
        state = np.sqrt([0.2, 0.5, 0.3]).astype(complex)
        result = self.assert_probability_from_oracle(sim, state)
        self.assertAlmostEqual(result.probability_optimum, 0.2, places=15)
        self.assertAlmostEqual(float((np.abs(state) ** 2)[old_mask].sum()), 0.7)
        # The correction changes classification, not the normalized-gap metric.
        self.assertEqual(result.normalized_gap,
                         float((np.abs(state) ** 2) @ sim.normalized_costs))

    def test_representably_near_degenerate_costs_remain_distinct(self):
        spacing = np.spacing(1.0)
        sim = self.simulator([0, spacing, 4 * spacing], q=np.eye(3))
        exact = exact_feasible_costs(sim.problem)
        self.assertEqual(exact[(0, 1, 0)] - exact[(1, 0, 0)],
                         Fraction.from_float(spacing))
        self.assertEqual(sim.cost_status, "nonconstant")
        self.assertEqual(sim.feasible_span, 4 * spacing)
        self.assert_probability_from_oracle(sim, sim.uniform_state)
        self.assertAlmostEqual(sim.evaluate([], []).probability_optimum, 1 / 3)

    def test_normalization_underflow_does_not_create_an_extra_optimum(self):
        with np.errstate(under="ignore"):
            sim = self.simulator([0, 1e-300, 1e100])
        exact = exact_feasible_costs(sim.problem)
        self.assertGreater(exact[(0, 1, 0)], exact[(1, 0, 0)])
        # A zero normalized value need not denote minimum cost: division can
        # underflow after the accurately summed costs have resolved the gap.
        np.testing.assert_array_equal(sim.normalized_costs, [0.0, 0.0, 1.0])
        state = np.array([0.0, 1.0, 0.0], dtype=complex)
        result = self.assert_probability_from_oracle(sim, state)
        self.assertEqual(result.probability_optimum, 0.0)
        self.assertEqual(result.normalized_gap, 0.0)
        self.assertEqual(result.cost_status, "nonconstant")

    def test_final_cost_rounding_ties_follow_the_declared_float_objective(self):
        sim = self.simulator([0, 0.125, 4], q=1e16 * np.eye(3))
        exact = exact_feasible_costs(sim.problem)
        values = [exact[tuple(state)] for state in sim.basis]
        self.assertEqual(sum(value == min(values) for value in values), 1)
        # The contract is the compensated float64 objective used for
        # normalization, not arbitrary-precision minimization of its inputs.
        # A final rounding tie is distinct from adding an optimality tolerance.
        rounded_values = np.array([float(value) for value in values])
        np.testing.assert_array_equal(rounded_values, [1e16, 1e16, 1e16 + 4])
        expected = rounded_values == min(rounded_values)
        np.testing.assert_array_equal(sim.optimal_mask, expected)
        self.assertEqual(sim.feasible_span, 4.0)
        self.assertEqual(sim.cost_status, "nonconstant")
        self.assertAlmostEqual(sim.evaluate([], []).probability_optimum, 2 / 3)

    def test_positive_objective_scaling_preserves_optimum_probability(self):
        coefficients = np.array([0, 1, 1 + 2.0**-36, 4])
        reference = self.simulator(coefficients, k=2)
        u, beta = [0.73], [0.31]
        expected = self.assert_probability_from_oracle(
            reference, reference.state_dimensionless(u, beta)
        )
        for exponent in (-600, -60, 0, 60, 500):
            with self.subTest(exponent=exponent):
                sim = self.simulator(np.ldexp(coefficients, exponent), k=2)
                result = self.assert_probability_from_oracle(
                    sim, sim.state_dimensionless(u, beta)
                )
                self.assertAlmostEqual(result.probability_optimum,
                                       expected.probability_optimum, places=14)
                self.assertAlmostEqual(result.normalized_gap, expected.normalized_gap,
                                       places=14)

    def test_additive_objective_shift_preserves_optimum_probability(self):
        coefficients = np.array([0, 1, 1 + 2.0**-36, 4])
        reference = self.simulator(coefficients, k=2)
        u, beta = [0.73], [0.31]
        exact_reference = exact_feasible_costs(reference.problem)
        expected = self.assert_probability_from_oracle(
            reference, reference.state_dimensionless(u, beta)
        )
        for shift in (-16.0, -0.5, 0.0, 0.5, 16.0):
            with self.subTest(shift=shift):
                sim = self.simulator(coefficients + shift / 2, k=2)
                exact_shifted = exact_feasible_costs(sim.problem)
                for state, value in exact_reference.items():
                    self.assertEqual(exact_shifted[state] - value,
                                     Fraction.from_float(shift))
                result = self.assert_probability_from_oracle(
                    sim, sim.state_dimensionless(u, beta)
                )
                self.assertAlmostEqual(result.probability_optimum,
                                       expected.probability_optimum, places=14)
                self.assertAlmostEqual(result.normalized_gap, expected.normalized_gap,
                                       places=14)

    def test_constant_and_unresolved_objectives_withhold_probability(self):
        fixtures = [
            ("constant", self.simulator([3, 3, 3, 3], k=2)),
            ("unresolved", self.simulator([0, 0.125, 0.25, 0.5], k=2,
                                         q=1e16 * np.eye(4))),
        ]
        for label, sim in fixtures:
            with self.subTest(kind=label):
                exact_costs = exact_feasible_costs(sim.problem)
                self.assertEqual(len(set(exact_costs.values())) == 1, label == "constant")
                self.assertEqual(sim.cost_status, "constant_or_unresolved")
                self.assertFalse(sim.optimal_mask.any())
                for result in (sim.evaluate([0.1], [0.2]),
                               sim.evaluate_dimensionless([0.1], [0.2])):
                    self.assertIsNone(result.probability_optimum)
                    self.assertIsNone(result.normalized_gap)
                    serialized = json.loads(json.dumps(asdict(result), allow_nan=False))
                    self.assertIsNone(serialized["probability_optimum"])

    def test_probability_normalization_and_feasibility_remain_preserved(self):
        for initialization in ("uniform", "mixer_low", "mixer_high"):
            with self.subTest(initialization=initialization):
                sim = self.simulator([0, 1, 1 + 2.0**-36, 4], k=2,
                                     initialization=initialization)
                np.testing.assert_array_equal(sim.basis.sum(axis=1),
                                              np.full(sim.feasible_dimension, 2))
                for state in (sim.state([0.3], [0.2]),
                              sim.state_dimensionless([0.3], [0.2])):
                    result = self.assert_probability_from_oracle(sim, state)
                    self.assertAlmostEqual(float((np.abs(state) ** 2).sum()), 1.0,
                                           places=14)
                    self.assertAlmostEqual(result.state_norm, 1.0, places=14)
                    self.assertAlmostEqual(result.feasibility_probability, 1.0,
                                           places=14)
                    self.assertGreaterEqual(result.probability_optimum, 0.0)
                    self.assertLessEqual(result.probability_optimum, 1.0)


if __name__ == "__main__":
    unittest.main()
