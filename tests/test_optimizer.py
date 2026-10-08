from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from struct2circuit.mixers import ring_mixer  # noqa: E402
from struct2circuit.optimize import optimize_p1  # noqa: E402
from struct2circuit.problems import CardinalityQUBO, block_correlated_qubo  # noqa: E402
from struct2circuit.simulator import FeasibleSubspaceQAOA  # noqa: E402


def historical_optimizer(simulator, grid_size):
    """Frozen raw-coordinate search from the pre-PR9 optimizer (f3121e2).

    This reference intentionally has no new coordinate or status handling.
    It protects numerical reproduction, including grid ties and solver options.
    """
    evaluations = 0

    def objective(theta):
        nonlocal evaluations
        evaluations += 1
        return simulator.evaluate([theta[0]], [theta[1]]).expectation

    points = [
        (objective((gamma, beta)), np.asarray([gamma, beta]))
        for gamma in np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False)
        for beta in np.linspace(0.0, np.pi, grid_size, endpoint=False)
    ]
    value, start = min(points, key=lambda item: item[0])
    candidates = [(value, start)]
    for method, options in (
        ("L-BFGS-B", {"maxiter": 120, "ftol": 1e-13, "gtol": 1e-9}),
        ("Powell", {"maxiter": 180, "ftol": 1e-12, "xtol": 1e-10}),
    ):
        result = minimize(
            objective,
            start.copy(),
            method=method,
            bounds=[(0.0, 2.0 * np.pi), (0.0, np.pi)],
            options=options,
        )
        if np.isfinite(result.fun):
            candidates.append((float(result.fun), result.x))
    _, selected = min(candidates, key=lambda item: item[0])
    return selected, simulator.evaluate([selected[0]], [selected[1]]), evaluations


class CoordinateSimulator:
    """Analytic objective for search-bound and candidate-provenance tests."""

    cost_status = "nonconstant"
    feasible_span = 100.0

    def __init__(self):
        self.calls = []

    def evaluate_dimensionless(self, u, beta):
        self.calls.append((float(u[0]), float(beta[0])))
        value = 0.1 + (u[0] - 1.1) ** 2 + (beta[0] - 1.2) ** 2
        return SimpleNamespace(
            expectation=20.0 + self.feasible_span * value,
            normalized_gap=float(value),
        )

    def evaluate(self, gamma, beta):
        raise AssertionError("feasible_span search must use normalized phases and objective")


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(314159)
        q = rng.normal(size=(6, 6))
        self.problem = CardinalityQUBO((q + q.T) / 2.0, rng.normal(size=6), 3)
        self.mixer = ring_mixer(6)

    def simulator(self, scale=1.0, shift=0.0):
        return FeasibleSubspaceQAOA(
            CardinalityQUBO(
                scale * self.problem.Q,
                scale * self.problem.c + shift / self.problem.k,
                self.problem.k,
            ),
            self.mixer,
        )

    def test_grid_and_both_refinements_share_dimensionless_bounds(self):
        simulator = CoordinateSimulator()
        calls = []

        def keep_start(objective, start, **kwargs):
            calls.append((kwargs["method"], tuple(kwargs["bounds"]), start.copy()))
            return SimpleNamespace(fun=objective(start), x=start, success=True)

        with patch("struct2circuit.optimize.minimize", side_effect=keep_start):
            result = optimize_p1(simulator, grid_size=5)

        bounds = ((0.0, 2.0 * np.pi), (0.0, np.pi))
        self.assertEqual([call[1] for call in calls], [bounds, bounds])
        np.testing.assert_array_equal(calls[0][2], calls[1][2])
        expected_grid = [
            (u, beta)
            for u in np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
            for beta in np.linspace(0.0, np.pi, 5, endpoint=False)
        ]
        np.testing.assert_array_equal(simulator.calls[:25], expected_grid)
        self.assertEqual(result.parameter_bounds, bounds)
        self.assertEqual(result.gamma_coordinate, "u")
        self.assertEqual(result.gamma[0], calls[0][2][0] / simulator.feasible_span)
        self.assertEqual(result.objective_evaluations, 27)
        self.assertEqual(len(simulator.calls), 28)  # final reporting call is excluded
        self.assertEqual(result.selected_source, "grid")  # grid wins exact ties
        self.assertIsNone(result.selected_source_success)
        self.assertFalse(result.local_refinement_converged)
        self.assertEqual(result.improvement_over_grid, 0.0)
        self.assertFalse(result.gamma_boundary_hit)
        self.assertFalse(result.beta_boundary_hit)

    def test_selected_candidate_provenance(self):
        # Every returned fun agrees with the analytic simulator at returned x.
        cases = [
            ("grid", [2.0, 2.0], True, [3.0, 2.0], True, None),
            ("lbfgsb", [1.1, 1.2], True, [1.2, 1.2], False, True),
            ("lbfgsb", [1.1, 1.2], False, [1.2, 1.2], True, False),
            ("powell", [1.2, 1.2], True, [1.1, 1.2], True, True),
        ]
        for source, lb_point, lb_success, pow_point, pow_success, selected_success in cases:
            with self.subTest(source=source, selected_success=selected_success):
                simulator = CoordinateSimulator()
                points = iter([(lb_point, lb_success), (pow_point, pow_success)])

                def candidate(objective, start, **kwargs):
                    point, success = next(points)
                    point = np.asarray(point)
                    return SimpleNamespace(fun=objective(point), x=point, success=success)

                with patch("struct2circuit.optimize.minimize", side_effect=candidate):
                    result = optimize_p1(simulator, grid_size=5)
                self.assertEqual(result.selected_source, source)
                self.assertIs(result.selected_source_success, selected_success)
                self.assertEqual(result.lbfgsb_success, lb_success)
                self.assertEqual(result.powell_success, pow_success)
                self.assertEqual(result.local_refinement_converged, bool(selected_success))
                self.assertTrue(result.finite_objective)
                self.assertGreaterEqual(result.improvement_over_grid, 0.0)

    def test_boundary_flags_and_finite_objective_are_independent_of_success(self):
        simulator = CoordinateSimulator()

        def boundary_candidate(objective, start, **kwargs):
            point = np.asarray([2.0 * np.pi, 0.0])
            return SimpleNamespace(fun=-1.0, x=point, success=False)

        with patch("struct2circuit.optimize.minimize", side_effect=boundary_candidate):
            result = optimize_p1(simulator, grid_size=5)
        self.assertTrue(result.gamma_boundary_hit)
        self.assertTrue(result.beta_boundary_hit)
        self.assertTrue(result.finite_objective)
        self.assertFalse(result.selected_source_success)
        self.assertFalse(result.local_refinement_converged)

        # A finite claimed solver fun must not override a nonfinite final metric.
        original = simulator.evaluate_dimensionless

        def nonfinite_final(u, beta):
            output = original(u, beta)
            output.expectation = float("nan")
            return output

        simulator.evaluate_dimensionless = nonfinite_final
        with patch("struct2circuit.optimize.minimize", side_effect=boundary_candidate):
            result = optimize_p1(simulator, grid_size=5)
        self.assertFalse(result.finite_objective)

    def test_positive_cost_rescaling_preserves_states_optima_and_optimized_gap(self):
        base = self.simulator()
        reference = optimize_p1(base, grid_size=7)
        reference_probabilities = abs(base.state_dimensionless([1.234], [0.723])) ** 2
        for scale in (1e-16, 1e-12, 0.01, 7.0, 100.0, 1e12):
            with self.subTest(scale=scale):
                simulator = self.simulator(scale=scale)
                result = optimize_p1(simulator, grid_size=7)
                np.testing.assert_array_equal(simulator.optimal_mask, base.optimal_mask)
                np.testing.assert_allclose(
                    abs(simulator.state_dimensionless([1.234], [0.723])) ** 2,
                    reference_probabilities,
                    rtol=0.0,
                    atol=1e-13,
                )
                self.assertAlmostEqual(
                    result.result.normalized_gap, reference.result.normalized_gap, places=10
                )
                self.assertAlmostEqual(
                    result.gamma[0] * simulator.feasible_span,
                    reference.gamma[0] * base.feasible_span,
                    places=6,
                )
                # Returned physical gamma describes the normalized-phase state
                # up to the irrelevant phase from subtracting C_min.
                np.testing.assert_allclose(
                    abs(simulator.state(result.gamma, result.beta)) ** 2,
                    abs(simulator.state_dimensionless(
                        [result.gamma[0] * simulator.feasible_span], result.beta
                    )) ** 2,
                    rtol=0.0,
                    atol=1e-12,
                )

    def test_additive_shifts_preserve_normalized_optimization(self):
        base = self.simulator()
        reference = optimize_p1(base, grid_size=7)
        for shift in (-8.0, 4.0, 10000.0):
            with self.subTest(shift=shift):
                simulator = self.simulator(shift=shift)
                result = optimize_p1(simulator, grid_size=7)
                self.assertAlmostEqual(
                    result.result.normalized_gap, reference.result.normalized_gap, places=9
                )
                self.assertAlmostEqual(
                    result.result.expectation - shift, reference.result.expectation, places=8
                )
                np.testing.assert_array_equal(simulator.optimal_mask, base.optimal_mask)
                np.testing.assert_allclose(
                    abs(simulator.state_dimensionless([1.234], [0.723])) ** 2,
                    abs(base.state_dimensionless([1.234], [0.723])) ** 2,
                    rtol=0.0,
                    atol=1e-12,
                )

    def test_legacy_matches_original_search(self):
        # Includes non-unit spans and shifted costs; no new normalized evaluator
        # may be called in this historical path.
        for problem in (self.problem, block_correlated_qubo(6, 3, 9)):
            with self.subTest(problem=problem.name):
                simulator = FeasibleSubspaceQAOA(problem, self.mixer)
                expected, expected_result, evaluations = historical_optimizer(simulator, 7)
                with patch.object(
                    simulator, "evaluate_dimensionless", side_effect=AssertionError("legacy")
                ):
                    actual = optimize_p1(simulator, grid_size=7, gamma_scale="legacy")
                np.testing.assert_array_equal([actual.gamma[0], actual.beta[0]], expected)
                self.assertEqual(actual.result.expectation, expected_result.expectation)
                self.assertEqual(actual.objective_evaluations, evaluations)
                self.assertEqual(actual.gamma_coordinate, "gamma")
                self.assertEqual(actual.parameter_bounds, ((0.0, 2.0 * np.pi), (0.0, np.pi)))

    def test_constant_cost_skips_search_and_serializes_undefined_gap(self):
        simulator = FeasibleSubspaceQAOA(
            CardinalityQUBO(np.zeros((6, 6)), np.ones(6), 3), self.mixer
        )
        with patch("struct2circuit.optimize.minimize") as solver:
            result = optimize_p1(simulator, grid_size=5)
        solver.assert_not_called()
        self.assertIsNone(result.result.normalized_gap)
        self.assertEqual(result.result.cost_status, "constant_or_unresolved")
        self.assertEqual(result.selected_source, "not_run")
        self.assertIsNone(result.selected_source_success)
        self.assertIsNone(result.lbfgsb_success)
        self.assertIsNone(result.powell_success)
        self.assertIsNone(result.gamma_boundary_hit)
        self.assertIsNone(result.beta_boundary_hit)
        self.assertIsNone(result.improvement_over_grid)
        self.assertEqual(result.objective_evaluations, 0)
        self.assertTrue(result.finite_objective)
        serialized = json.loads(json.dumps(asdict(result), allow_nan=False))
        self.assertIsNone(serialized["result"]["normalized_gap"])

    def test_unrepresentable_returned_gamma_is_rejected(self):
        simulator = CoordinateSimulator()
        simulator.feasible_span = 1e-310

        def keep_start(objective, start, **kwargs):
            return SimpleNamespace(fun=objective(start), x=start, success=True)

        with patch("struct2circuit.optimize.minimize", side_effect=keep_start):
            with self.assertRaisesRegex(ValueError, "physical gamma"):
                optimize_p1(simulator, grid_size=5)


if __name__ == "__main__":
    unittest.main()
