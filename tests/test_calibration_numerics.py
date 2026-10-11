from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from struct2circuit.calibration.budget import Budget, BudgetExhausted, Limits
from struct2circuit.calibration.config import BY_NAME
from struct2circuit.calibration.diagnostics import (direct_oracle, gradient_check, invariance_checks,
    reuse_initialization, spectral_coordinates, spectral_diagnostics)
from struct2circuit.calibration.objective import Objective, production_search
from struct2circuit.calibration.reference import Candidate, ReferenceResult, ReferenceSearch
from struct2circuit.mixers import ring_mixer, complete_mixer
from struct2circuit.problems import CardinalityQUBO
from struct2circuit.simulator import FeasibleSubspaceQAOA


def fixture(initialization='uniform', constant=False):
    Q = np.diag([.1, .7, -.5, 1.])
    Q[0, 2] = Q[2, 0] = .23
    c = np.array([-.4, .1, .3, -.2])
    if constant: Q, c = np.zeros((4, 4)), np.ones(4)
    return FeasibleSubspaceQAOA(CardinalityQUBO(Q, c, 2), ring_mixer(4), initialization=initialization)


def budget(total=30000, landscape=30000):
    return Budget(Limits(essential=total, shared=total, total=total, landscape=landscape, optional=10, cpu_seconds=30))


class NumericalTests(unittest.TestCase):
    def test_native_objective_and_exact_coordinates(self):
        sim = fixture()
        obj = Objective(sim, budget(), 'toy')
        point = (.713456781234567, .391)
        evaluated = obj.evaluate(*point)
        native = sim.evaluate_dimensionless([point[0]], [point[1]])
        self.assertEqual(evaluated.gap, native.normalized_gap)
        self.assertEqual(evaluated.expectation, native.expectation)
        self.assertEqual(evaluated.u, point[0])
        self.assertEqual(evaluated.gamma, point[0]/sim.feasible_span)
        state = expm(-1j*point[1]*sim.mixer_hamiltonian)@expm(-1j*point[0]*np.diag(sim.normalized_costs))@sim.initial_state
        self.assertLess(np.max(abs(abs(state)**2-abs(evaluated.state)**2)), 1e-10)

    def test_raw_expectation_is_reporting_only(self):
        sim = fixture()
        original = sim._evaluate_state
        with patch.object(sim, '_evaluate_state', side_effect=lambda state: replace(original(state), expectation=-999.)):
            evaluated = Objective(sim, budget(), 'toy').evaluate(.3, .4)
        self.assertEqual(evaluated.expectation, -999.)
        self.assertGreaterEqual(evaluated.gap, 0)

    def test_gradient_and_direct_oracle_all_initializations(self):
        for init in ('uniform', 'mixer_low', 'mixer_high'):
            with self.subTest(initialization=init):
                obj = Objective(fixture(init), budget(), 'toy:'+init)
                self.assertEqual(gradient_check(obj, (.631, -.23), 'Bsign')['status'], 'pass')
                self.assertEqual(direct_oracle(obj, (.631, -.23), 'Bsign')['status'], 'pass')
                self.assertEqual(spectral_coordinates(obj, (.631, .23))['status'], 'pass')

    def test_oracle_mismatch_is_detected(self):
        obj = Objective(fixture(), budget(), 'toy')
        with patch('struct2circuit.calibration.diagnostics.expm', side_effect=lambda matrix: np.eye(len(matrix))):
            self.assertEqual(direct_oracle(obj, (.631, .43))['status'], 'fail')

    def test_invariance_and_fallback_are_separate(self):
        for init in ('uniform', 'mixer_low', 'mixer_high'):
            obj = Objective(fixture(init), budget(), 'toy:'+init)
            records = invariance_checks(obj, [(0., 0.), (.37, .23)])
            self.assertTrue(all(row['status'] == 'pass' for row in records))
            transport = [r for r in records if r['kind'] == 'cyclic_transport']
            reconstruction = [r for r in records if r['kind'] == 'cyclic_reconstruction']
            self.assertEqual(len(transport), 2)
            self.assertEqual(len(reconstruction), 2)
            self.assertTrue(all('selection_status' not in r for r in transport))
            self.assertTrue(all('selection_status' in r for r in reconstruction))
        base = fixture()
        degenerate = FeasibleSubspaceQAOA(base.problem, complete_mixer(4), initialization='mixer_low')
        spectrum = spectral_diagnostics(degenerate)
        self.assertGreater(spectrum['extrema']['low']['multiplicity'], 1)
        self.assertEqual(spectrum['extrema']['low']['projection_branch'], 'lexicographic_fallback')

    def test_eigensolver_reuse_retains_initialization(self):
        base, b = fixture(), budget()
        reused = reuse_initialization(base, 'mixer_high', b)
        self.assertIs(reused._mixer_eigenvectors, base._mixer_eigenvectors)
        np.testing.assert_array_equal(reused.initial_state, base.mixer_high_state)
        self.assertEqual(b.counts['eigensolver_reuse'], 1)

    def test_constant_is_not_searched(self):
        obj = Objective(fixture(constant=True), budget(), 'constant')
        search = ReferenceSearch(obj, 64, (1042, 0, 0))
        with patch('struct2circuit.calibration.reference.minimize', side_effect=AssertionError('No constant optimization')):
            result = search.search(BY_NAME['B00'])
        self.assertFalse(result.complete)
        self.assertEqual(result.unresolved_reason, 'constant_or_unresolved')
        self.assertEqual(obj.budget.counts['actual_evaluations'], 0)
        self.assertIsNone(obj.evaluate(0, 0).gap)

    def test_exact_cache_keys_and_fresh_reevaluation(self):
        b = budget()
        obj = Objective(fixture(), b, 'toy')
        obj.evaluate(.2, .3)
        obj.evaluate(.2, .3)
        self.assertEqual(b.counts['actual_evaluations'], 1)
        self.assertEqual(b.counts['cache_hits'], 1)
        obj.evaluate(np.nextafter(.2, np.inf), .3)
        obj.evaluate(.2, .3, fresh=True)
        self.assertEqual(b.counts['actual_evaluations'], 3)
        self.assertEqual(len(b.unique), 2)
        before = b.counts['actual_evaluations']
        direct_oracle(obj, (.2, .3))
        self.assertEqual(b.counts['actual_evaluations'], before+1)
        self.assertEqual(len(b.unique), 2)

    def test_production_recipe_preserved_and_u_recorded(self):
        obj = Objective(fixture(), budget(), 'toy')
        result, selected = production_search(obj, 9)
        self.assertEqual(selected.beta, result.beta[0])
        self.assertEqual(selected.gamma, result.gamma[0])
        self.assertEqual(selected.gap, result.result.normalized_gap)
        self.assertEqual(result.parameter_bounds, BY_NAME['B00'].bounds)
        with self.assertRaises(ValueError): production_search(obj, 11)

    def test_cache_does_not_merge_distinct_instance_identities(self):
        b, shared, sim = budget(), {}, fixture()
        first = Objective(sim, b, '1042:0:0', cache=shared)
        alias = Objective(sim, b, '1042:0:0', cache=shared)
        second = Objective(sim, b, '1043:0:0', cache=shared)
        first.evaluate(.2, .3)
        alias.evaluate(.2, .3)
        second.evaluate(.2, .3)
        self.assertEqual(b.counts['actual_evaluations'], 2)
        self.assertEqual(b.counts['cache_hits'], 1)
        self.assertEqual(len(b.unique), 2)

    def test_native_norm_error_is_independently_rejected(self):
        sim = fixture()
        original = sim.state_dimensionless
        with patch.object(sim, 'state_dimensionless', side_effect=lambda *args: original(*args)*1.000001):
            record = direct_oracle(Objective(sim, budget(), 'toy'), (.3, .2))
        self.assertEqual(record['status'], 'fail')
        self.assertGreater(record['errors']['native_normalization'], 1e-10)


class ReferenceTests(unittest.TestCase):
    def test_reference_routes_and_actual_synthetic_search(self):
        obj = Objective(fixture(), budget(), 'toy')
        search = ReferenceSearch(obj, 64, (1042, 0, 0))
        result = search.search(BY_NAME['B00'])
        self.assertIsNotNone(result.route_a)
        self.assertIsNotNone(result.route_b)
        self.assertTrue(BY_NAME['B00'].contains(result.selected.point))
        self.assertTrue(result.faces_complete)
        self.assertTrue(any(a['source'] == 'de_0' for a in result.attempts))
        self.assertTrue(any(a['source'] == 'de_polish_0' for a in result.attempts))
        de = next(a for a in result.attempts if a['source'] == 'de_0')
        self.assertLessEqual(de['objective_calls'], 1024)
        self.assertLess(obj.budget.counts['actual_evaluations'], 10000)
        for c in result.candidates: self.assertTrue(BY_NAME['B00'].contains(c.point))

    def test_external_pool_cannot_manufacture_route_agreement(self):
        obj = Objective(fixture(), budget(), 'toy')
        search = ReferenceSearch(obj, 64, (1042, 0, 0))
        points = [(.3, .2), (.4, .2), (.5, .2)]
        candidates = [Candidate(*p, obj.evaluate(*p).gap, route, 'test', objective_identity=obj.identity)
                      for p, route in zip(points, ('A', 'B', 'A'))]
        result = ReferenceResult('B00', candidates=candidates[:2], faces_complete=True)
        search._pool(result, BY_NAME['B00'], [candidates[2]])
        self.assertEqual(result.route_a, candidates[0].value)
        self.assertEqual(result.route_b, candidates[1].value)
        self.assertEqual(result.candidates[-1].route, 'pool')
        with self.assertRaises(ValueError): search._pool(result, BY_NAME['B00'], [replace(candidates[2], objective_identity='different')])

    def test_out_of_bounds_and_false_candidate_value_rejected(self):
        obj = Objective(fixture(), budget(), 'toy')
        search = ReferenceSearch(obj, 64, (1042, 0, 0))
        with self.assertRaises(ArithmeticError): search._candidate((-1, 0), BY_NAME['B00'], 'A', 'invalid')
        candidate = Candidate(.2, .3, -1., 'A', 'bad', objective_identity=obj.identity)
        with self.assertRaises(ArithmeticError): search._pool(ReferenceResult('B00'), BY_NAME['B00'], [candidate])

    def test_budget_interruption_preserves_grid_candidate(self):
        b = budget(total=3)
        result = ReferenceSearch(Objective(fixture(), b, 'toy'), 64, (1042, 0, 0)).search(BY_NAME['B00'])
        self.assertFalse(result.complete)
        self.assertIsNotNone(result.selected)
        self.assertEqual(len(result.candidates), 3)
        self.assertEqual(b.counts['actual_evaluations'], 3)

    def test_only_one_refinement_allowed(self):
        obj = Objective(fixture(), budget(total=1), 'toy')
        result = ReferenceResult('B00', pooled=.5)
        search = ReferenceSearch(obj, 64, (1042, 0, 0))
        search.refine(result, audit_only=True)
        with self.assertRaises(ValueError): search.refine(result)
        self.assertFalse(result.complete)

    def test_flat_beta_zero_and_spectral_u_zero(self):
        obj = Objective(fixture('mixer_high'), budget(), 'toy')
        result = ReferenceSearch(obj, 64, (1042, 0, 2)).search(BY_NAME['B00'])
        self.assertIn('1:0.0', result.flat_faces)
        self.assertIn('0:0.0', result.flat_faces)


if __name__ == '__main__': unittest.main()


class ReferenceContractTests(unittest.TestCase):
    def test_starts_use_exact_grid_indices_and_exempt_production(self):
        obj = Objective(fixture(), budget(), 'toy')
        search = ReferenceSearch(obj, 64, (1042, 0, 0))
        axes = (np.array([0., .1, .2, .3, .4]), np.array([0., .1, .2]))
        best = Candidate(0., 0., .1, 'A', 'grid')
        separated = Candidate(.2, 0., .2, 'A', 'grid')
        production = Candidate(.19, .01, .15, 'production', 'P9')
        starts = search._starts([best, separated], production, axes, 4)
        self.assertEqual(starts, [best, production, separated])

    def test_exact_de_options_and_separate_polishing(self):
        from types import SimpleNamespace
        from struct2circuit.calibration.config import de_seed
        obj = Objective(fixture(), budget(), 'toy')
        search = ReferenceSearch(obj, 64, (1042, 3, 1))
        result = ReferenceResult('B00')
        captured = []
        def fake_de(fun, bounds, **kwargs):
            captured.append((bounds, kwargs))
            point = np.array([.25, .35])
            value = fun(point)
            return SimpleNamespace(x=point, fun=value, success=True, message='synthetic backend')
        with patch('struct2circuit.calibration.reference.differential_evolution', side_effect=fake_de), patch.object(search, '_local') as polish:
            search._global(result, BY_NAME['B00'], 0)
            search._global(result, BY_NAME['B00'], 1)
        self.assertEqual(polish.call_count, 2)
        for replicate, (bounds, settings) in enumerate(captured):
            self.assertEqual(bounds, BY_NAME['B00'].bounds)
            self.assertEqual(settings['maxiter'], 63 if replicate == 0 else 126)
            self.assertEqual(settings['seed'], de_seed(1042, 3, 1, 0, replicate))
            self.assertEqual(settings['strategy'], 'best1bin')
            self.assertEqual(settings['popsize'], 8)
            self.assertEqual(settings['tol'], 1e-8)
            self.assertEqual(settings['atol'], 1e-10)
            self.assertEqual(settings['mutation'], (.5, 1))
            self.assertEqual(settings['recombination'], .7)
            self.assertEqual(settings['init'], 'latinhypercube')
            self.assertEqual(settings['updating'], 'immediate')
            self.assertEqual(settings['workers'], 1)
            self.assertFalse(settings['polish'])
            self.assertNotIn('x0', settings)

    def test_solver_attempt_cap_preserves_valid_observations(self):
        obj = Objective(fixture(), budget(), 'toy')
        search = ReferenceSearch(obj, 64, (1042, 0, 0))
        result = ReferenceResult('B00')
        def excessive(fun):
            for i in range(20): fun((.1+i*.01, .2))
            raise AssertionError('Attempt cap was bypassed')
        selected = search._attempt(result, BY_NAME['B00'], 'A', 'cap_test', excessive, 5)
        self.assertIsNotNone(selected)
        self.assertEqual(len(result.candidates), 5)
        self.assertEqual(result.attempts[0]['objective_calls'], 5)
        self.assertEqual(result.attempts[0]['interrupted'], 'AttemptLimit')

    def test_optional_bv_is_separate_and_budgeted(self):
        from struct2circuit.calibration.diagnostics import common_spectral_time_reference
        b = Budget(Limits(essential=5, shared=5, total=10, landscape=10, optional=2, cpu_seconds=10))
        obj = Objective(fixture(), b, 'toy', account='shared', optional=True)
        with self.assertRaises(ValueError): common_spectral_time_reference(obj, 3., 64, (1042, 0, 0))
        b.mandatory_complete = True
        record = common_spectral_time_reference(obj, 3., 64, (1042, 0, 0))
        self.assertFalse(record['completed'])
        self.assertEqual(record['production_qualification'], 'NOT_TESTED')
        self.assertEqual(record['domain'].name, 'Bv')
        self.assertEqual(b.counts['optional'], 2)


class BoundaryCutRegressionTests(unittest.TestCase):
    def analytic_search(self, axis=1, offset=.04):
        from types import SimpleNamespace
        from struct import pack
        class AnalyticObjective:
            identity = 'isolated-analytic-boundary'
            simulator = SimpleNamespace(initial_state=np.ones(2)/np.sqrt(2), mixer_hamiltonian=np.diag([0., 1.]))
            def __init__(self): self.budget = budget(); self.cache = {}
            def evaluate(self, u, beta, domain='B00', **kw):
                key = (self.identity, pack('>dd', u, beta))
                self.budget.request('analytic', domain)
                if key not in self.cache:
                    self.budget.charge(key, 'analytic', domain)
                    target = (2*np.pi if axis == 0 else np.pi)+offset
                    self.cache[key] = SimpleNamespace(gap=((u,beta)[axis]-target)**2/40)
                return self.cache[key]
        obj = AnalyticObjective()
        return ReferenceSearch(obj, 64, (1042,0,0))

    def test_both_sides_and_exact_internal_cuts(self):
        from struct2circuit.calibration.config import grid_axes
        from scipy.optimize import minimize_scalar
        for axis in (0,1):
            for offset in (-.04,0.,.04):
                with self.subTest(axis=axis, offset=offset):
                    search = self.analytic_search(axis, offset); result = ReferenceResult('B11'); brackets = []
                    def spy(fun, **kw):
                        brackets.append(kw['bounds']); return minimize_scalar(fun, **kw)
                    with patch('struct2circuit.calibration.reference.minimize_scalar', side_effect=spy):
                        search._faces(result, BY_NAME['B11'], grid_axes(BY_NAME['B11'],64), 0)
                    target = (2*np.pi if axis == 0 else np.pi)+offset
                    self.assertTrue(result.faces_complete)
                    self.assertLess(result.selected.value, 1e-12)
                    self.assertAlmostEqual(result.selected.point[axis], target, delta=2e-7)
                    cut = 2*np.pi if axis == 0 else np.pi
                    self.assertTrue(any(hi == cut for lo,hi in brackets))
                    self.assertTrue(any(lo == cut for lo,hi in brackets))
                    self.assertTrue(any(c.point[axis] == cut for c in result.candidates))

    def test_original_right_of_pi_trace_recovered(self):
        from struct2circuit.calibration.config import grid_axes
        search = self.analytic_search(); result = ReferenceResult('B01')
        search._faces(result, BY_NAME['B01'], grid_axes(BY_NAME['B01'],64),0)
        self.assertLess(result.selected.value, 1e-12)
        self.assertAlmostEqual(result.selected.beta, np.pi+.04, delta=2e-7)
        self.assertTrue(result.faces_complete)

    def test_second_subbracket_budget_interruption_is_incomplete(self):
        from struct2circuit.calibration.config import grid_axes
        from scipy.optimize import minimize_scalar
        search = self.analytic_search(); result = ReferenceResult('B01'); visited = []
        def interrupted(fun, **kw):
            visited.append(kw['bounds'])
            if kw['bounds'][0] == np.pi:
                count = search.objective.budget.counts['actual_evaluations']
                search.objective.budget.limits = replace(search.objective.budget.limits, total=count+1)
            return minimize_scalar(fun, **kw)
        with patch('struct2circuit.calibration.reference.minimize_scalar', side_effect=interrupted):
            with self.assertRaises(BudgetExhausted):
                search._faces(result, BY_NAME['B01'], grid_axes(BY_NAME['B01'],64),0)
        self.assertTrue(any(lo == np.pi for lo,hi in visited))
        self.assertFalse(result.faces_complete)
        self.assertFalse(result.complete)
        self.assertTrue(result.candidates)
        self.assertEqual(search.objective.budget.counts['actual_evaluations'], search.objective.budget.limits.total)

    def test_flat_trace_keeps_endpoints_without_scalar_work(self):
        from struct2circuit.calibration.config import grid_axes
        search = self.analytic_search()
        def flat(*args, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(gap=.25)
        search.objective.evaluate = flat
        result = ReferenceResult('B11')
        with patch('struct2circuit.calibration.reference.minimize_scalar', side_effect=AssertionError('Flat trace')):
            search._faces(result, BY_NAME['B11'], grid_axes(BY_NAME['B11'],64),0)
        self.assertTrue(result.faces_complete)
        self.assertTrue(all(any(c.point == p for c in result.candidates) for p in ((0.,0.),(4*np.pi,2*np.pi))))

    def test_per_bracket_and_trace_caps_remain_unresolved(self):
        from struct2circuit.calibration.config import grid_axes
        search = self.analytic_search(); result = ReferenceResult('B11'); calls = []
        def capped(fun, **kw):
            calls.append(0)
            lo,hi = kw['bounds']
            for i in range(61):
                calls[-1] += 1
                fun(lo+(hi-lo)*(i+1)/100)
        with patch('struct2circuit.calibration.reference.minimize_scalar', side_effect=capped):
            search._faces(result, BY_NAME['B11'], grid_axes(BY_NAME['B11'],64),0)
        self.assertFalse(result.faces_complete)
        self.assertTrue(calls)
        self.assertLessEqual(max(calls),61)  # 61st request raises before evaluation.

    def test_both_internal_coordinate_cuts_are_partitioned_in_one_scan(self):
        from types import SimpleNamespace
        from struct2circuit.calibration.config import grid_axes
        from scipy.optimize import minimize_scalar
        search=self.analytic_search();original=search.objective.evaluate;brackets=[]
        def evaluate(u,beta,*args,**kwargs):
            value=original(u,beta,*args,**kwargs)
            return SimpleNamespace(gap=(value.gap+(u-(2*np.pi+.04))**2/40)/2)
        search.objective.evaluate=evaluate
        def scan(fun,**kw):
            brackets.append(kw['bounds']);return minimize_scalar(fun,**kw)
        result=ReferenceResult('B11')
        with patch('struct2circuit.calibration.reference.minimize_scalar',side_effect=scan):
            search._faces(result,BY_NAME['B11'],grid_axes(BY_NAME['B11'],64),0)
        self.assertTrue(result.faces_complete)
        for cut in (np.pi,2*np.pi):
            self.assertTrue(any(hi==cut for lo,hi in brackets))
            self.assertTrue(any(lo==cut for lo,hi in brackets))
            self.assertFalse(any(lo<cut<hi for lo,hi in brackets))
