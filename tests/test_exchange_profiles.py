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

from struct2circuit.mixers import (
    exchange_profile_scores,
    exchange_profile_statistics,
    score_conditioned_mixer,
)
from struct2circuit.problems import CardinalityQUBO


class ExchangeProfileTests(unittest.TestCase):
    @staticmethod
    def diagonal_transfer_fixture(exponent: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(101719)
        raw = rng.integers(-8, 9, (7, 7))
        q = (raw + raw.T).astype(float) * 2.0**exponent
        np.fill_diagonal(q, 0.0)
        return q, np.zeros(7)

    @staticmethod
    def exact_feasible_costs(problem: CardinalityQUBO) -> dict[tuple, Fraction]:
        costs = {}
        for state in problem.feasible_basis():
            occupied = np.flatnonzero(state)
            costs[tuple(state)] = (
                sum(Fraction.from_float(float(problem.Q[i, j]))
                    for i in occupied for j in occupied)
                + sum(Fraction.from_float(float(problem.c[i])) for i in occupied)
            )
        return costs

    def assert_moments_match_exact_costs(self, problem: CardinalityQUBO, costs: dict) -> None:
        moments = exchange_profile_statistics(problem.Q, problem.c, problem.k)
        np.testing.assert_array_equal(moments.mean, -moments.mean.T)
        np.testing.assert_array_equal(moments.variance, moments.variance.T)
        np.testing.assert_array_equal(moments.rms, moments.rms.T)
        for i, j in combinations(range(problem.n), 2):
            changes = []
            for state, cost in costs.items():
                if state[i] == 1 and state[j] == 0:
                    swapped = list(state)
                    swapped[i], swapped[j] = 0, 1
                    changes.append(costs[tuple(swapped)] - cost)
            mean = sum(changes) / len(changes)
            variance = sum((change - mean)**2 for change in changes) / len(changes)
            rms = sqrt(float(sum(change**2 for change in changes) / len(changes)))
            scale = float(max(abs(change) for change in changes))
            # Scale-relative absolute allowances protect zero means without
            # letting the tiny resolved fixtures pass with all-zero moments.
            for actual, expected, unit in (
                (moments.mean[i, j], float(mean), scale),
                (moments.variance[i, j], float(variance), scale**2),
                (moments.rms[i, j], rms, scale),
            ):
                np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=5e-14 * unit)

    @staticmethod
    def large_offset_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Exact adversarial-review fixture: all 21 original RMS scores differ.
        rng = np.random.default_rng(290009)
        raw = rng.integers(-3, 4, size=(7, 7))
        q = (raw + raw.T).astype(float)
        c = rng.integers(-3, 4, size=7).astype(float)
        v = 1e14 * np.array([0., 1., -1., 1., -1., 0., 1.])
        return q, c, v

    def assert_moments_match_enumeration(self, problem: CardinalityQUBO) -> None:
        moments = exchange_profile_statistics(problem.Q, problem.c, problem.k)
        basis = problem.feasible_basis()
        np.testing.assert_array_equal(moments.mean, -moments.mean.T)
        np.testing.assert_array_equal(moments.variance, moments.variance.T)
        np.testing.assert_array_equal(moments.rms, moments.rms.T)
        for values in (moments.mean, moments.variance, moments.rms):
            np.testing.assert_array_equal(np.diag(values), 0.0)
        for i, j in combinations(range(problem.n), 2):
            states = basis[(basis[:, i] == 1) & (basis[:, j] == 0)]
            swapped = states.copy()
            swapped[:, i], swapped[:, j] = 0, 1
            deltas = problem.costs(swapped) - problem.costs(states)
            with self.subTest(n=problem.n, k=problem.k, i=i, j=j):
                np.testing.assert_allclose(
                    [moments.mean[i, j], moments.variance[i, j], moments.rms[i, j]],
                    [np.mean(deltas), np.var(deltas), np.sqrt(np.mean(deltas**2))],
                    rtol=5e-13,
                    atol=5e-13,
                )

    def test_exchange_formula_matches_every_small_feasible_swap(self) -> None:
        rng = np.random.default_rng(12903)
        for n in range(2, 8):
            q = rng.normal(size=(n, n))
            q = (q + q.T) / 2.0
            c = rng.normal(size=n)
            for k in range(1, n):
                problem = CardinalityQUBO(q, c, k)
                basis = problem.feasible_basis()
                for i, j in combinations(range(n), 2):
                    states = basis[(basis[:, i] == 1) & (basis[:, j] == 0)]
                    swapped = states.copy()
                    swapped[:, i], swapped[:, j] = 0, 1
                    others = [ell for ell in range(n) if ell not in (i, j)]
                    expected = (
                        q[j, j] - q[i, i] + c[j] - c[i]
                        + 2.0 * states[:, others] @ (q[j, others] - q[i, others])
                    )
                    deltas = problem.costs(swapped) - problem.costs(states)
                    with self.subTest(n=n, k=k, i=i, j=j):
                        np.testing.assert_allclose(deltas, expected, rtol=5e-13, atol=5e-13)
                        reverse = (
                            q[i, i] - q[j, j] + c[i] - c[j]
                            + 2.0 * swapped[:, others] @ (q[i, others] - q[j, others])
                        )
                        np.testing.assert_allclose(reverse, -deltas, rtol=5e-13, atol=5e-13)
                        changed = q.copy()
                        changed[i, j] += 7.0
                        changed[j, i] += 7.0
                        alternative = CardinalityQUBO(changed, c, k)
                        np.testing.assert_allclose(
                            alternative.costs(swapped) - alternative.costs(states), deltas,
                            rtol=5e-13, atol=5e-13,
                        )

    def test_conditional_moments_match_exhaustive_small_systems(self) -> None:
        rng = np.random.default_rng(19021)
        for n in range(2, 8):
            raw = rng.normal(size=(n, n))
            q = (raw + raw.T) / 2.0
            c = rng.normal(size=n)
            row_offsets = rng.normal(size=n)
            families = {
                "random": (q, c),
                "diagonal": (np.diag(rng.normal(size=n)), np.zeros(n)),
                "linear": (np.zeros((n, n)), c),
                "equal_coefficients": (
                    (row_offsets[:, None] + row_offsets[None, :]) / 2.0, c,
                ),
                "zero": (np.zeros((n, n)), np.zeros(n)),
            }
            for family, (matrix, linear) in families.items():
                for k in range(1, n):
                    with self.subTest(family=family, n=n, k=k):
                        self.assert_moments_match_enumeration(CardinalityQUBO(matrix, linear, k))

    def test_empty_and_single_remaining_site_have_zero_variance(self) -> None:
        for n in (2, 3):
            q = np.arange(n * n, dtype=float).reshape(n, n)
            q = (q + q.T) / 2.0
            c = np.arange(n, dtype=float)
            for k in range(1, n):
                moments = exchange_profile_statistics(q, c, k)
                np.testing.assert_array_equal(moments.variance, 0.0)
                np.testing.assert_array_equal(moments.rms, np.abs(moments.mean))
                self.assert_moments_match_enumeration(CardinalityQUBO(q, c, k))

    def test_endpoint_cardinalities_have_exactly_zero_variance(self) -> None:
        rng = np.random.default_rng(810)
        q = rng.normal(size=(7, 7))
        q = (q + q.T) / 2.0
        c = rng.normal(size=7)
        for k in (1, 6):
            np.testing.assert_array_equal(exchange_profile_statistics(q, c, k).variance, 0.0)

    @staticmethod
    def endpoint_cancellation_fixture(
        large: float = 1e16, residual: float = 1.0,
    ) -> CardinalityQUBO:
        q = np.zeros((5, 5))
        q[1, 2] = q[2, 1] = large
        q[1, 3] = q[3, 1] = residual
        c = np.zeros(5)
        c[1] = -2.0 * large
        return CardinalityQUBO(q, c, 4)

    @staticmethod
    def exact_conditional_changes(costs: dict, i: int, j: int) -> list[Fraction]:
        changes = []
        for state, cost in costs.items():
            if state[i] == 1 and state[j] == 0:
                swapped = list(state)
                swapped[i], swapped[j] = 0, 1
                changes.append(costs[tuple(swapped)] - cost)
        return changes

    def assert_endpoint_residual(self, problem: CardinalityQUBO, expected: Fraction) -> None:
        costs = self.exact_feasible_costs(problem)
        changes = self.exact_conditional_changes(costs, 0, 1)
        self.assertEqual(changes, [expected])
        moments = exchange_profile_statistics(problem.Q, problem.c, problem.k)
        self.assertEqual(moments.mean[0, 1], float(expected))
        self.assertEqual(moments.variance[0, 1], 0.0)
        self.assertEqual(moments.rms[0, 1], float(abs(expected)))
        self.assert_moments_match_exact_costs(problem, costs)

    def test_endpoint_mean_preserves_large_cancellation_residual(self) -> None:
        self.assert_endpoint_residual(self.endpoint_cancellation_fixture(), Fraction(2))

    def test_endpoint_mean_preserves_unit_scale_cancellation_residual(self) -> None:
        self.assert_endpoint_residual(
            self.endpoint_cancellation_fixture(1.0, 2.0**-54), Fraction(1, 2**53),
        )

    def endpoint_topology_fixture(self) -> tuple[CardinalityQUBO, np.ndarray]:
        problem = self.endpoint_cancellation_fixture()
        problem = CardinalityQUBO(
            problem.Q, np.array([0., -2e16, -2e16 + 8., 22., 17.]), problem.k,
        )
        costs = self.exact_feasible_costs(problem)
        exact_scores = np.zeros((problem.n, problem.n))
        for i, j in combinations(range(problem.n), 2):
            changes = self.exact_conditional_changes(costs, i, j)
            self.assertEqual(len(changes), 1)
            exact_scores[i, j] = exact_scores[j, i] = float(abs(changes[0]))
        # Verify the independently enumerated oracle against the review fixture.
        levels = np.array([0., 2., 8., 24., 17.])
        np.testing.assert_array_equal(exact_scores, np.abs(levels[:, None] - levels[None, :]))
        self.assertEqual(len(np.unique(exact_scores[np.triu_indices(5, 1)])), 10)
        return problem, exact_scores

    def test_endpoint_distinct_scores_preserve_exact_low_topology(self) -> None:
        problem, exact_scores = self.endpoint_topology_fixture()
        scores = exchange_profile_scores(problem.Q, problem.c, problem.k)
        np.testing.assert_array_equal(scores, exact_scores)
        self.assertEqual(len(np.unique(scores[np.triu_indices(5, 1)])), 10)
        self.assertEqual(
            score_conditioned_mixer(scores, 6, prefer="low").edges,
            ((0, 1), (0, 2), (1, 2), (1, 4), (2, 4), (3, 4)),
        )

    def test_endpoint_cancellation_scores_and_graph_are_permutation_equivariant(self) -> None:
        problem, exact_scores = self.endpoint_topology_fixture()
        permutation = np.array([0, 1, 3, 2, 4])
        indexing = np.ix_(permutation, permutation)
        scores = exchange_profile_scores(problem.Q[indexing], problem.c[permutation], problem.k)
        inverse = np.argsort(permutation)
        restored_scores = scores[np.ix_(inverse, inverse)]
        np.testing.assert_array_equal(restored_scores, exact_scores)
        np.testing.assert_array_equal(
            restored_scores, exchange_profile_scores(problem.Q, problem.c, problem.k),
        )
        self.assertEqual(restored_scores[2, 3], 16.0)
        restored_edges = tuple(sorted(
            tuple(sorted((permutation[i], permutation[j])))
            for i, j in score_conditioned_mixer(scores, 6, prefer="low").edges
        ))
        self.assertEqual(restored_edges, ((0, 1), (0, 2), (1, 2), (1, 4), (2, 4), (3, 4)))

    def test_ordinary_cardinality_mean_preserves_cancellation_residuals(self) -> None:
        # Include a nonintegral 2*m/N: rounding a row sum or its weighted
        # contribution before cancellation also loses represented information.
        for n, k, large, residual, linear in (
            (6, 3, 1e16, 1.0, -1e16),
            (5, 2, 3e16, 3.0, -2e16),
            (5, 2, 3.0, 3.0 * 2.0**-54, -2.0),
            (5, 2, 1.5e16, 1.0, -1e16),
            (5, 2, 1.5, 2.0**-54, -1.0),
        ):
            with self.subTest(n=n, k=k, large=large):
                q = np.zeros((n, n))
                q[1, 2] = q[2, 1] = large
                q[1, 3] = q[3, 1] = residual
                c = np.zeros(n)
                c[1] = linear
                problem = CardinalityQUBO(q, c, k)
                costs = self.exact_feasible_costs(problem)
                changes = self.exact_conditional_changes(costs, 0, 1)
                expected_mean = sum(changes) / len(changes)
                self.assertGreater(expected_mean, 0)
                actual = exchange_profile_statistics(q, c, k)
                self.assertEqual(actual.mean[0, 1], float(expected_mean))
                self.assert_moments_match_exact_costs(problem, costs)

    def test_cancelling_coefficients_match_exact_costs_at_all_cardinalities(self) -> None:
        rng = np.random.default_rng(984113)
        for n in range(2, 8):
            raw = rng.integers(-8, 9, (n, n))
            q = (raw + raw.T).astype(float) * 2.0**-20
            # These diagonal/linear terms cancel in the exact represented
            # objective; the off-diagonal residuals remain in the inputs.
            transfer = 1e12 * rng.integers(-4, 5, n)
            np.fill_diagonal(q, transfer)
            c = -transfer
            for k in range(1, n):
                with self.subTest(n=n, k=k):
                    problem = CardinalityQUBO(q, c, k)
                    self.assert_moments_match_exact_costs(problem, self.exact_feasible_costs(problem))

    def test_equal_coefficients_preserve_mean_and_have_zero_variance(self) -> None:
        q = np.zeros((5, 5))
        q[1, 2:] = q[2:, 1] = 2.0
        c = np.array([0.0, -3.0, 0.0, 0.0, 0.0])
        for k in range(1, 5):
            moments = exchange_profile_statistics(q, c, k)
            self.assertEqual(moments.mean[0, 1], -3.0 + 4.0 * (k - 1))
            self.assertEqual(moments.variance[0, 1], 0.0)
            self.assertEqual(moments.rms[0, 1], abs(moments.mean[0, 1]))

    def test_cancelling_exchange_has_zero_rms_despite_nonzero_coefficients(self) -> None:
        q = np.zeros((4, 4))
        q[1, 2:] = q[2:, 1] = 1.0
        c = np.array([0.0, -2.0, 0.0, 0.0])
        moments = exchange_profile_statistics(q, c, 2)
        self.assertEqual(moments.mean[0, 1], 0.0)
        self.assertEqual(moments.variance[0, 1], 0.0)
        self.assertEqual(moments.rms[0, 1], 0.0)
        np.testing.assert_array_equal(exchange_profile_scores(q, c, 2), moments.rms)

    def test_positive_scaling_of_mean_variance_and_rms(self) -> None:
        rng = np.random.default_rng(916)
        q = rng.normal(size=(6, 6))
        q = (q + q.T) / 2.0
        c = rng.normal(size=6)
        reference = exchange_profile_statistics(q, c, 3)
        for scale in (1e-12, 1e-6, 1.0, 7.0, 1e6, 1e12):
            with self.subTest(scale=scale):
                actual = exchange_profile_statistics(scale * q, scale * c, 3)
                np.testing.assert_allclose(actual.mean / scale, reference.mean, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(
                    actual.variance / scale**2, reference.variance, rtol=1e-12, atol=1e-12,
                )
                np.testing.assert_allclose(actual.rms / scale, reference.rms, rtol=1e-12, atol=1e-12)

    def test_additive_constants_and_equivalent_encodings_preserve_moments(self) -> None:
        rng = np.random.default_rng(448)
        n = 6
        q = rng.normal(size=(n, n))
        q = (q + q.T) / 2.0
        c = rng.normal(size=n)
        v = np.arange(n, dtype=float)
        for k in range(1, n):
            reference = exchange_profile_statistics(q, c, k)
            variants = {
                "linear_constant": (q, c + 7.0 / k, 7.0),
                "diagonal_constant": (q + 7.0 / k * np.eye(n), c, 7.0),
                "quadratic_constant": (q + 7.0 / k**2 * np.ones((n, n)), c, 7.0),
                "diagonal_linear_transfer": (q + np.diag(v), c - v, 0.0),
                "feasible_null_term": (q + (v[:, None] + v[None, :]) / 2.0, c - k * v, 0.0),
            }
            problem = CardinalityQUBO(q, c, k)
            basis = problem.feasible_basis()
            for name, (matrix, linear, constant) in variants.items():
                with self.subTest(k=k, encoding=name):
                    np.testing.assert_allclose(
                        CardinalityQUBO(matrix, linear, k).costs(basis),
                        problem.costs(basis) + constant, rtol=1e-12, atol=1e-12,
                    )
                    actual = exchange_profile_statistics(matrix, linear, k)
                    for field in ("mean", "variance", "rms"):
                        np.testing.assert_allclose(
                            getattr(actual, field), getattr(reference, field),
                            rtol=1e-12, atol=1e-12,
                        )

    def test_feasible_null_objective_has_zero_exchange_moments(self) -> None:
        n, k = 6, 3
        v = np.arange(n, dtype=float)
        q = (v[:, None] + v[None, :]) / 2.0
        c = -k * v
        problem = CardinalityQUBO(q, c, k)
        np.testing.assert_array_equal(problem.costs(problem.feasible_basis()), 0.0)
        moments = exchange_profile_statistics(q, c, k)
        for values in (moments.mean, moments.variance, moments.rms):
            np.testing.assert_array_equal(values, 0.0)

    def test_large_equivalent_encoding_preserves_moments_and_ranked_graphs(self) -> None:
        q, c, v = self.large_offset_fixture()
        n, k = 7, 3
        original = CardinalityQUBO(q, c, k)
        equivalent = CardinalityQUBO(q + v[:, None] + v[None, :], c - 2 * k * v, k)
        basis = original.feasible_basis()
        np.testing.assert_array_equal(original.costs(basis), equivalent.costs(basis))
        reference = exchange_profile_statistics(q, c, k)
        actual = exchange_profile_statistics(equivalent.Q, equivalent.c, k)
        for field in ("mean", "variance", "rms"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(reference, field))
        self.assert_moments_match_enumeration(equivalent)

        indices = np.triu_indices(n, k=1)
        self.assertEqual(len(np.unique(reference.rms[indices])), 21)
        np.testing.assert_array_equal(
            np.argsort(actual.rms[indices]), np.argsort(reference.rms[indices]),
        )
        expected_graphs = {
            "low": ((0, 2), (0, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (3, 4)),
            "high": ((0, 1), (0, 6), (1, 2), (1, 4), (2, 6), (3, 6), (4, 5), (4, 6)),
        }
        for prefer, expected in expected_graphs.items():
            with self.subTest(prefer=prefer):
                self.assertEqual(score_conditioned_mixer(reference.rms, 8, prefer=prefer).edges, expected)
                self.assertEqual(score_conditioned_mixer(actual.rms, 8, prefer=prefer).edges, expected)

    def test_large_equivalent_encoding_preserves_scaling_and_permutations(self) -> None:
        q, c, v = self.large_offset_fixture()
        k = 3
        reference = exchange_profile_statistics(q, c, k)
        equivalent_q = q + v[:, None] + v[None, :]
        equivalent_c = c - 2 * k * v
        rng = np.random.default_rng(8194)
        permutations = [np.arange(7), *(rng.permutation(7) for _ in range(5))]
        # Binary scales preserve the represented input coefficients exactly,
        # isolating arithmetic invariance from information lost at input time.
        for scale in (2.0**-20, 1.0, 2.0**20):
            for permutation in permutations:
                with self.subTest(scale=scale, permutation=permutation):
                    indexing = np.ix_(permutation, permutation)
                    actual = exchange_profile_statistics(
                        scale * equivalent_q[indexing], scale * equivalent_c[permutation], k,
                    )
                    for field, factor in (("mean", scale), ("variance", scale**2), ("rms", scale)):
                        np.testing.assert_allclose(
                            getattr(actual, field) / factor, getattr(reference, field)[indexing],
                            rtol=2e-14, atol=2e-14,
                        )
                    for prefer in ("low", "high"):
                        edges = score_conditioned_mixer(actual.rms, 8, prefer=prefer).edges
                        restored_edges = tuple(sorted(
                            (min(permutation[i], permutation[j]), max(permutation[i], permutation[j]))
                            for i, j in edges
                        ))
                        self.assertEqual(
                            restored_edges, score_conditioned_mixer(reference.rms, 8, prefer=prefer).edges,
                        )

    def test_diagonal_transfer_preserves_resolved_moments_and_both_graphs(self) -> None:
        for exponent, magnitude in ((-20, 1e12), (-60, 1.0)):
            q, c = self.diagonal_transfer_fixture(exponent)
            original = CardinalityQUBO(q, c, 3)
            costs = self.exact_feasible_costs(original)
            reference = exchange_profile_statistics(q, c, 3)
            edge_indices = np.triu_indices(7, 1)
            self.assertEqual(len(np.unique(reference.rms[edge_indices])), 21)
            self.assertGreater(float(np.max(reference.rms)), 0.0)
            for pattern in (np.ones(7), np.array([1., -2., 3., -4., 0., 2., -1.])):
                with self.subTest(exponent=exponent, pattern=pattern):
                    transfer = magnitude * pattern
                    equivalent = CardinalityQUBO(q + np.diag(transfer), c - transfer, 3)
                    exact_equivalent = self.exact_feasible_costs(equivalent)
                    self.assertEqual(exact_equivalent, costs)
                    actual = exchange_profile_statistics(equivalent.Q, equivalent.c, 3)
                    for field in ("mean", "variance", "rms"):
                        np.testing.assert_array_equal(getattr(actual, field), getattr(reference, field))
                    self.assertEqual(len(np.unique(actual.rms[edge_indices])), 21)
                    self.assert_moments_match_exact_costs(equivalent, exact_equivalent)
                    for prefer in ("low", "high"):
                        self.assertEqual(
                            score_conditioned_mixer(actual.rms, 8, prefer=prefer).edges,
                            score_conditioned_mixer(reference.rms, 8, prefer=prefer).edges,
                        )

    def test_diagonal_transfer_preserves_scaling_and_permutations(self) -> None:
        rng = np.random.default_rng(120923)
        for exponent, magnitude in ((-20, 1e12), (-60, 1.0)):
            q, c = self.diagonal_transfer_fixture(exponent)
            transfer = magnitude * np.array([1., -2., 3., -4., 0., 2., -1.])
            reference = exchange_profile_statistics(q, c, 3)
            for scale in (2.0**-20, 1.0, 2.0**20):
                for _ in range(3):
                    permutation = rng.permutation(7)
                    indexing = np.ix_(permutation, permutation)
                    actual = exchange_profile_statistics(
                        scale * (q + np.diag(transfer))[indexing],
                        scale * (c - transfer)[permutation], 3,
                    )
                    for field, factor in (("mean", scale), ("variance", scale**2), ("rms", scale)):
                        np.testing.assert_allclose(
                            getattr(actual, field) / factor, getattr(reference, field)[indexing],
                            rtol=5e-14, atol=0.0,
                        )
                    for prefer in ("low", "high"):
                        restored_edges = tuple(sorted(
                            tuple(sorted((permutation[i], permutation[j])))
                            for i, j in score_conditioned_mixer(actual.rms, 8, prefer=prefer).edges
                        ))
                        self.assertEqual(
                            restored_edges, score_conditioned_mixer(reference.rms, 8, prefer=prefer).edges,
                        )

    def test_diagonal_transfers_match_exact_enumeration_at_all_cardinalities(self) -> None:
        rng = np.random.default_rng(721031)
        for n in range(2, 8):
            raw = rng.integers(-8, 9, (n, n))
            q = (raw + raw.T).astype(float) * 2.0**-20
            np.fill_diagonal(q, 0.0)
            c = np.zeros(n)
            transfer = 1e12 * rng.integers(-4, 5, n)
            for k in range(1, n):
                with self.subTest(n=n, k=k):
                    original = CardinalityQUBO(q, c, k)
                    equivalent = CardinalityQUBO(q + np.diag(transfer), c - transfer, k)
                    costs = self.exact_feasible_costs(original)
                    self.assertEqual(self.exact_feasible_costs(equivalent), costs)
                    self.assert_moments_match_exact_costs(equivalent, costs)

    def test_moments_are_permutation_equivariant(self) -> None:
        rng = np.random.default_rng(197)
        n, k = 7, 3
        q = rng.normal(size=(n, n))
        q = (q + q.T) / 2.0
        c = rng.normal(size=n)
        reference = exchange_profile_statistics(q, c, k)
        edge_scores = reference.rms[np.triu_indices(n, k=1)]
        self.assertEqual(len(np.unique(edge_scores)), len(edge_scores))
        for _ in range(10):
            permutation = rng.permutation(n)
            indexing = np.ix_(permutation, permutation)
            actual = exchange_profile_statistics(q[indexing], c[permutation], k)
            for field in ("mean", "variance", "rms"):
                np.testing.assert_allclose(
                    getattr(actual, field), getattr(reference, field)[indexing],
                    rtol=1e-12, atol=1e-12,
                )

    def test_invalid_problem_inputs_are_rejected(self) -> None:
        q, c = np.zeros((4, 4)), np.zeros(4)
        for k in (0, 4, -1, 1.5, True, np.bool_(True)):
            with self.subTest(k=k), self.assertRaises(ValueError):
                exchange_profile_statistics(q, c, k)
        inputs = [
            (np.zeros((3, 4)), np.zeros(3), 1),
            (np.zeros((1, 1)), np.zeros(1), 1),
            (q, np.zeros(3), 1),
            (np.full((4, 4), np.inf), c, 1),
            (q, np.full(4, np.nan), 1),
            (np.triu(np.ones((4, 4))), c, 1),
        ]
        for matrix, linear, k in inputs:
            with self.subTest(matrix=matrix, linear=linear), self.assertRaises(ValueError):
                exchange_profile_statistics(matrix, linear, k)


if __name__ == "__main__":
    unittest.main()
