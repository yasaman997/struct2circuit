from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import sqrt
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from struct2circuit.mixers import exchange_profile_statistics
from struct2circuit.problems import (
    CardinalityQUBO,
    block_correlated_qubo,
    weak_structure_null_qubo,
    weighted_densest_k_subgraph_qubo,
    weighted_max_k_vertex_cover_qubo,
)


class ProblemSymmetryTests(unittest.TestCase):
    def assert_both_accept(self, q: np.ndarray, k: int = 2) -> CardinalityQUBO:
        original = q.copy()
        problem = CardinalityQUBO(q, np.zeros(len(q)), k)
        moments = exchange_profile_statistics(problem.Q, problem.c, problem.k)
        np.testing.assert_array_equal(q, original)
        np.testing.assert_array_equal(problem.Q, original)
        for values in (moments.mean, moments.variance, moments.rms):
            self.assertTrue(np.isfinite(values).all())
        return problem

    def assert_both_reject(self, q: np.ndarray) -> None:
        c = np.zeros(len(q))
        for operation in (
            lambda: CardinalityQUBO(q, c, 2),
            lambda: exchange_profile_statistics(q, c, 2),
        ):
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "symmetric"):
                operation()

    def test_exactly_symmetric_matrices_are_accepted(self) -> None:
        q = np.array([[1., 2., -3.], [2., 4., 5.], [-3., 5., 6.]])
        self.assert_both_accept(q)

    def test_large_symmetric_coefficients_are_accepted(self) -> None:
        q = np.array([[1., 2., -3.], [2., 4., 5.], [-3., 5., 6.]]) * 1e100
        self.assert_both_accept(q)

    def test_large_asymmetry_previously_hidden_by_relative_tolerance_is_rejected(self) -> None:
        q = np.full((3, 3), 1e12)
        q[0, 1] += 1e5
        # The original constructor admitted this matrix, while the descriptor
        # rejected it. This is an absolute discrepancy of 100,000 cost units.
        self.assertTrue(np.allclose(q, q.T, atol=1e-12))
        self.assert_both_reject(q)

    def test_absolute_tolerance_boundary_is_shared(self) -> None:
        for discrepancy in (2.0**-41, 1e-12):
            with self.subTest(discrepancy=discrepancy):
                q = np.zeros((3, 3))
                q[0, 1] = discrepancy
                self.assert_both_accept(q)
        for discrepancy in (np.nextafter(1e-12, np.inf), 2.0**-39):
            with self.subTest(discrepancy=discrepancy):
                q = np.zeros((3, 3))
                q[0, 1] = discrepancy
                self.assert_both_reject(q)

    def test_near_symmetry_preserves_costs_and_has_bounded_descriptor_error(self) -> None:
        n, k = 5, 3
        epsilon = 2.0**-41
        q = np.zeros((n, n))
        q[0, 2:] = epsilon
        q[1, 2:] = -epsilon
        problem = self.assert_both_accept(q, k)
        self.assertFalse(np.array_equal(problem.Q, problem.Q.T))

        # Independent oracle: both original triangles enter every feasible
        # cost. No symmetric-Q exchange formula is used to obtain the changes.
        exact_costs = {}
        for state in problem.feasible_basis():
            occupied = np.flatnonzero(state)
            exact_costs[tuple(state)] = sum(
                Fraction.from_float(float(q[i, j]))
                for i in occupied for j in occupied
            )
        np.testing.assert_array_equal(
            problem.costs(problem.feasible_basis()),
            [float(value) for value in exact_costs.values()],
        )
        moments = exchange_profile_statistics(problem.Q, problem.c, k)
        bound = 2 * (k - 1) * epsilon
        exact_means = {}
        for i, j in combinations(range(n), 2):
            changes = []
            for state, cost in exact_costs.items():
                if state[i] and not state[j]:
                    swapped = list(state)
                    swapped[i], swapped[j] = 0, 1
                    changes.append(exact_costs[tuple(swapped)] - cost)
            mean = float(sum(changes) / len(changes))
            rms = sqrt(float(sum(value**2 for value in changes) / len(changes)))
            exact_means[i, j] = mean
            self.assertLessEqual(abs(moments.mean[i, j] - mean), bound)
            self.assertLessEqual(abs(moments.rms[i, j] - rms), bound)
        # The error bound is attained here: accepted input noise need not be
        # negligible relative to a small objective and does not imply exactness.
        self.assertEqual(exact_means[0, 1], -2 * (k - 1) * epsilon)
        self.assertEqual(moments.mean[0, 1], -4 * (k - 1) * epsilon)
        self.assertEqual(abs(moments.mean[0, 1] - exact_means[0, 1]), bound)

    def test_existing_generators_remain_symmetric_and_deterministic(self) -> None:
        generators = (
            lambda: block_correlated_qubo(7, 3, 9),
            lambda: weighted_densest_k_subgraph_qubo(7, 3, 0.5, 4),
            lambda: weighted_max_k_vertex_cover_qubo(7, 3, 0.5, 4),
            lambda: weak_structure_null_qubo(7, 3, 0.5, 1.2, 4),
        )
        for generate in generators:
            first, second = generate(), generate()
            with self.subTest(generator=first.name):
                np.testing.assert_array_equal(first.Q, first.Q.T)
                np.testing.assert_array_equal(first.Q, second.Q)
                np.testing.assert_array_equal(first.c, second.c)
                self.assertEqual(first.metadata, second.metadata)
                exchange_profile_statistics(first.Q, first.c, first.k)


if __name__ == "__main__":
    unittest.main()
