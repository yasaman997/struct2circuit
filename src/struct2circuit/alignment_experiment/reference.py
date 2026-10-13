"""Small B00-only specialization of reviewed reference and oracle primitives."""
from itertools import product
from math import pi

import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

from ..calibration.budget import BudgetExhausted
from ..calibration.config import BY_NAME
from ..calibration.diagnostics import direct_oracle, gradient_check
from ..calibration.objective import Objective as NativeObjective
from ..calibration.reference import ReferenceSearch as NativeSearch, AttemptLimit
from ..mixers import MixerSpec
from ..problems import CardinalityQUBO
from ..simulator import FeasibleSubspaceQAOA
from .config import seed


class Objective(NativeObjective):
    def gradient(self, point, domain):
        self.evaluate(*point, domain)
        # d_u psi requires a separate mixer evolution even at a cached objective.
        self.budget.charge(self.key(*point), self.landscape, domain, derivative=True)
        return super().gradient(point, domain)


def interior(family, replicate, width):
    margin = np.array((1e-5, 1e-5/max(1., width)))
    return np.random.Generator(np.random.PCG64(seed('controls', family, replicate))).uniform(
        margin, np.array((2*pi, pi))-margin, size=(8, 2))


class ReferenceSearch(NativeSearch):
    def __init__(self, objective, family, replicate):
        super().__init__(objective, 128, (0, 0, 0))
        self.family, self.replicate = family, replicate
        self.face_records = []

    def _global(self, result, domain, replicate):
        value = seed(f'de{replicate}', self.family, self.replicate)
        rng = np.random.Generator(np.random.PCG64(value))
        candidate = self._attempt(result, domain, 'B', f'de_{replicate}',
            lambda fun: differential_evolution(fun, domain.bounds, seed=rng, strategy='best1bin',
                popsize=8, maxiter=63 if replicate == 0 else 126, tol=1e-8, atol=1e-10,
                mutation=(.5, 1), recombination=.7, init='latinhypercube', updating='immediate',
                workers=1, polish=False), 1024 if replicate == 0 else 2032, value)
        if candidate:
            self._local(result, domain, candidate.point, route='B', source=f'de_polish_{replicate}', seed=value)

    def _faces(self, result, domain, axes, level):
        if domain.name != 'B00':
            raise ValueError('Mechanism references require B00-only faces')
        complete = True
        for axis in (0, 1):
            for fixed in domain.bounds[axis]:
                trace = [self._candidate((fixed, float(x)) if axis == 0 else (float(x), fixed),
                         domain, 'A', 'face_grid') for x in axes[1-axis]]
                result.candidates.extend(trace)
                flat = axis == 1 and fixed == 0
                if axis == 0 and fixed == 0:
                    sim = self.objective.simulator
                    state = sim.initial_state
                    eigenvalue = np.vdot(state, sim.mixer_hamiltonian@state).real
                    flat = (2*pi*np.linalg.norm(sim.mixer_hamiltonian@state-eigenvalue*state) <= 1e-10
                            and np.ptp([c.value for c in trace]) <= 1e-10)
                fresh_start = self.objective.budget.counts['actual_evaluations']
                done = True
                brackets = []
                if not flat:
                    for i in range(1, len(trace)-1):
                        if trace[i].value >= min(trace[i-1].value, trace[i+1].value):
                            continue
                        calls = 0
                        def fun(x):
                            nonlocal calls
                            if calls >= 60 or self.objective.budget.counts['actual_evaluations']-fresh_start >= 150:
                                raise AttemptLimit('boundary_face_cap')
                            calls += 1
                            point = (fixed, float(x)) if axis == 0 else (float(x), fixed)
                            c = self._candidate(point, domain, 'A', 'face_scalar')
                            result.candidates.append(c)
                            return c.value
                        try:
                            solved = minimize_scalar(fun, bounds=(trace[i-1].point[1-axis], trace[i+1].point[1-axis]),
                                                     method='bounded', options=dict(xatol=1e-10, maxiter=60))
                            done = done and bool(solved.success)
                            brackets.append(dict(bounds=[trace[i-1].point[1-axis], trace[i+1].point[1-axis]],
                                                 calls=calls, complete=bool(solved.success)))
                        except AttemptLimit:
                            brackets.append(dict(bounds=[trace[i-1].point[1-axis], trace[i+1].point[1-axis]],
                                                 calls=calls, complete=False))
                            done = False
                            break
                complete = complete and done
                self.face_records.append(dict(level=level, axis=axis, fixed=fixed, complete=done,
                                              flat=bool(flat), points=[c.point for c in trace],
                                              values=[c.value for c in trace], brackets=brackets))
        result.faces_complete = complete

    def _bundle(self, result, domain):
        if result.full_bundle:
            raise ValueError('A second bundle is prohibited')
        result.full_bundle = True
        self._global(result, domain, 1)
        distinct = {}
        for c in sorted(result.candidates, key=lambda c: (c.value, c.u, c.beta)):
            distinct.setdefault(c.point, c)
        for start in list(distinct.values())[:2]:
            self._attempt(result, domain, start.route, 'powell',
                lambda fun, p=start.point: minimize(fun, np.asarray(p), method='Powell', bounds=domain.bounds,
                    options=dict(maxiter=180, ftol=1e-12, xtol=1e-10)), 250)
            result.attempts[-1]['start'] = start.point

    def coarse(self):
        return self.search(BY_NAME['B00'])

    def fine(self, result):
        if self.objective.budget.stop_reason or self.objective.landscape in self.objective.budget.arm_stops:
            result.complete = False
            return result
        return self.refine(result, audit_only=False)


def pre_controls(obj, family, replicate):
    points = interior(family, replicate, float(np.ptp(obj.simulator._mixer_eigenvalues)))
    for p in points:
        yield gradient_check(obj, p, 'B00')
    for p in [*points, *product((0., 2*pi), (0., pi))]:
        yield direct_oracle(obj, p, 'B00')


def transport_controls(obj):
    """Joint cyclic transport only; no cost-scale or wider-domain diagnostics."""
    sim = obj.simulator
    p = (np.arange(6)+1) % 6  # old vertex -> new vertex
    inverse = np.argsort(p)
    problem = CardinalityQUBO(sim.problem.Q[np.ix_(inverse, inverse)], sim.problem.c[inverse], 3)
    graph = MixerSpec('transport', 6, tuple((int(p[i]), int(p[j])) for i, j in sim.mixer.edges), 'joint transport')
    other_sim = FeasibleSubspaceQAOA(problem, graph, initialization='uniform')
    index = {tuple(x): i for i, x in enumerate(sim.basis)}
    mapping = [index[tuple(x[p])] for x in other_sim.basis]
    if not np.array_equal(other_sim.mixer_hamiltonian, sim.mixer_hamiltonian[np.ix_(mapping, mapping)]):
        raise ArithmeticError('Transported Hamiltonian differs')
    other = Objective(other_sim, obj.budget, obj.landscape)
    for point in product(np.linspace(0., 2*pi, 5), np.linspace(0., pi, 5)):
        a, b = obj.evaluate(*point), other.evaluate(*point)
        error = max(abs(a.gap-b.gap), float(np.max(abs(abs(a.state[mapping])**2-abs(b.state)**2))))
        yield dict(kind='cyclic_transport', domain='B00', point=point, error=error,
                   status='pass' if error <= 1e-10 else 'fail', transported_identity=other.identity)


def final_controls(obj, result):
    winners = [min((c for c in result.candidates if c.route == route), key=lambda c: (c.value, c.u, c.beta))
               for route in ('A', 'B')]
    winners.append(result.selected)
    for c in winners:
        checked = obj.evaluate(*c.point, fresh=True)
        error = abs(checked.gap-c.value)
        yield dict(kind='candidate_reevaluation', domain='B00', point=c.point, error=error,
                   value=c.value, checked_value=checked.gap, status='pass' if error <= 1e-10 else 'fail')
        yield direct_oracle(obj, c.point, 'B00')
