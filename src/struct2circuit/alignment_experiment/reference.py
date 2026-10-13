"""Small B00-only specialization of reviewed reference and oracle primitives."""
from itertools import product
from math import pi
from dataclasses import dataclass, asdict, replace

import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

from ..calibration.budget import BudgetExhausted
from ..calibration.config import BY_NAME, numerical_settings
from ..calibration.diagnostics import gradient_check as native_gradient_check
from ..calibration.objective import Objective as NativeObjective, Evaluation as NativeEvaluation, array_hash
from ..calibration.reference import ReferenceSearch as NativeSearch, AttemptLimit, Candidate as NativeCandidate
from ..calibration.storage import canonical_hash
from scipy.linalg import expm
from ..mixers import MixerSpec
from ..problems import CardinalityQUBO
from ..simulator import FeasibleSubspaceQAOA
from .config import seed


@dataclass(frozen=True)
class Evaluation(NativeEvaluation):
    computation_id: str


@dataclass(frozen=True)
class Candidate(NativeCandidate):
    computation_id: str | None = None
    request_id: str | None = None
    attempt_id: str | None = None
    observation: str = 'search'
    candidate_id: str | None = None


def candidate_id(record):
    return canonical_hash({k: v for k, v in record.items() if k != 'candidate_id'})


class Objective(NativeObjective):
    def evaluate(self, u, beta, domain='B00', *, fresh=False):
        if self.released:
            raise RuntimeError('Numerical work attempted after terminal cache release')
        key = self.key(u, beta)
        self.last_request_id, self.last_request_completed = None, False
        request = self.budget.request(self.landscape, domain, key=key, fresh=fresh)
        self.last_request_id = request
        if not fresh and key in self.cache:
            self.last = self.cache[key]
            self.budget.resolve(request, self.last.computation_id, cached=True)
            self.last_request_completed = True
            return self.last
        computation = None
        try:
            computation = self.budget.charge(key, self.landscape, domain, request_id=request, fresh=fresh)
            state = self.simulator.state_dimensionless([u], [beta])
            result = self.simulator._evaluate_state(state)
            span = self.simulator.feasible_span
            gamma = float(u)/span if span > 0 else None
            if gamma is not None and (not np.isfinite(gamma) or (u != 0 and gamma == 0)):
                raise ArithmeticError('Physical gamma is outside the supported float64 range')
            if not np.isfinite(state).all() or result.normalized_gap is not None and not np.isfinite(result.normalized_gap):
                raise ArithmeticError('Nonfinite normalized state/objective')
            state.setflags(write=False)
            value = Evaluation(float(u), float(beta), gamma, result.normalized_gap, result.expectation,
                span, result.cost_status, state, float(np.ptp(self.simulator._mixer_eigenvalues)), computation)
            self.budget.outcome(computation, dict(gap=value.gap, state_sha256=array_hash(state)))
            self.budget.resolve(request, computation)
        except BaseException as error:
            if computation:
                self.budget.outcome(computation, error=error)
            self.budget.resolve(request, computation, error=error)
            raise
        self.cache[key] = self.last = value
        self.last_request_completed = True
        self.budget.check_time()
        return value

    def gradient(self, point, domain):
        self.evaluate(*point, domain)
        # d_u psi requires a separate mixer evolution even at a cached objective.
        computation = self.budget.charge(self.key(*point), self.landscape, domain, derivative=True)
        started = self.budget.counts['gradient_computations']
        try:
            result = super().gradient(point, domain)
            if not np.isfinite(result).all():
                raise ArithmeticError('Nonfinite analytic derivative')
            self.budget.outcome(computation, result.tolist(),
                gradient_started=self.budget.counts['gradient_computations']>started)
            return result
        except BaseException as error:
            self.budget.outcome(computation, error=error,
                gradient_started=self.budget.counts['gradient_computations']>started)
            raise


def interior(family, replicate, width):
    margin = np.array((1e-5, 1e-5/max(1., width)))
    return np.random.Generator(np.random.PCG64(seed('controls', family, replicate))).uniform(
        margin, np.array((2*pi, pi))-margin, size=(8, 2))


class ReferenceSearch(NativeSearch):
    def __init__(self, objective, family, replicate):
        super().__init__(objective, 128, (0, 0, 0))
        self.family, self.replicate = family, replicate
        self.face_records = []
        self.active_attempt = None
        self.observed_candidates = []

    def _candidate(self, point, domain, route, source, *, fresh=False, seed=None, success=None):
        c = super()._candidate(point, domain, route, source, fresh=fresh, seed=seed, success=success)
        c = Candidate(**asdict(c), computation_id=self.objective.last.computation_id,
            request_id=self.objective.last_request_id, attempt_id=self.active_attempt,
            observation='terminal' if fresh else 'solver' if self.active_attempt else 'search')
        c = replace(c, candidate_id=candidate_id(asdict(c)))
        self.observed_candidates.append(c)
        return c

    def _attempt(self, result, domain, route, source, runner, cap, seed=None, *, start=None, start_candidate=None):
        settings = numerical_settings()['de' if source in ('de_0', 'de_1') else 'powell' if source == 'powell' else 'local']
        spec = dict(objective_identity=self.objective.identity, index=len(result.attempts), route=route,
            source=source, seed=seed, start=None if start is None else tuple(map(float, start)),
            start_candidate_id=start_candidate, bounds=domain.bounds, settings=settings, objective_call_cap=cap)
        attempt = dict(spec, attempt_id=canonical_hash(spec), observations=[], objective_calls=0,
            terminal_candidate_id=None, success=False, interrupted=None, message=None, failed_request_id=None)
        attempt.update(unreturned_request_id=None, blocked_call=False)
        result.attempts.append(attempt)  # preserve partial attempts on every exception
        self.partial_result = result
        start_actual = self.objective.budget.counts['actual_evaluations']
        start_cpu = self.objective.budget.cpu_clock()
        self.active_attempt = attempt['attempt_id']
        observed = []
        def fun(point):
            if attempt['objective_calls'] >= cap:
                raise AttemptLimit('solver_objective_call_cap')
            attempt['objective_calls'] += 1
            try:
                c = self._candidate(point, domain, route, source, seed=seed)
            except BaseException:
                request = getattr(self.objective, 'last_request_id', None)
                if request is None:
                    attempt['blocked_call'] = True
                elif getattr(self.objective, 'last_request_completed', False):
                    attempt['unreturned_request_id'] = request
                else:
                    attempt['failed_request_id'] = request
                raise
            observed.append(c); result.candidates.append(c)
            attempt['observations'].append(c.candidate_id)
            return c.value
        with self.objective.budget.trace() as work:
            try:
                solved = runner(fun)
                if not domain.contains(solved.x):
                    raise ArithmeticError('Solver returned an out-of-domain candidate')
                c = self._candidate(solved.x, domain, route, source, fresh=True, seed=seed, success=bool(solved.success))
                observed.append(c); result.candidates.append(c)
                attempt['terminal_candidate_id'] = c.candidate_id
                attempt.update(success=bool(solved.success), message=str(solved.message))
            except BaseException as error:
                attempt.update(interrupted=type(error).__name__, message=str(error))
                if not isinstance(error, AttemptLimit):
                    raise
            finally:
                attempt.update(actual_evaluations=self.objective.budget.counts['actual_evaluations']-start_actual,
                    cpu_seconds=self.objective.budget.cpu_clock()-start_cpu, work=work)
                self.active_attempt = None
        return min(observed, key=lambda c: (c.value, c.u, c.beta)) if observed else None

    def _local(self, result, domain, start, route='A', source='lbfgsb', seed=None, *, start_candidate=None):
        return self._attempt(result, domain, route, source,
            lambda fun: minimize(fun, np.asarray(start), method='L-BFGS-B',
                jac=lambda x: self.objective.gradient(x, domain.name), bounds=domain.bounds,
                options=dict(maxiter=120, ftol=1e-13, gtol=1e-9)), 250, seed,
            start=start, start_candidate=start_candidate)

    def _global(self, result, domain, replicate):
        value = seed(f'de{replicate}', self.family, self.replicate)
        rng = np.random.Generator(np.random.PCG64(value))
        candidate = self._attempt(result, domain, 'B', f'de_{replicate}',
            lambda fun: differential_evolution(fun, domain.bounds, seed=rng, strategy='best1bin',
                popsize=8, maxiter=63 if replicate == 0 else 126, tol=1e-8, atol=1e-10,
                mutation=(.5, 1), recombination=.7, init='latinhypercube', updating='immediate',
                workers=1, polish=False), 1024 if replicate == 0 else 2032, value)
        if candidate:
            self._local(result, domain, candidate.point, route='B', source=f'de_polish_{replicate}', seed=value,
                        start_candidate=candidate.candidate_id)

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
                        observations = []
                        def fun(x):
                            nonlocal calls
                            if calls >= 60 or self.objective.budget.counts['actual_evaluations']-fresh_start >= 150:
                                raise AttemptLimit('boundary_face_cap')
                            calls += 1
                            point = (fixed, float(x)) if axis == 0 else (float(x), fixed)
                            c = self._candidate(point, domain, 'A', 'face_scalar')
                            result.candidates.append(c)
                            observations.append(c.candidate_id)
                            return c.value
                        try:
                            solved = minimize_scalar(fun, bounds=(trace[i-1].point[1-axis], trace[i+1].point[1-axis]),
                                                     method='bounded', options=dict(xatol=1e-10, maxiter=60))
                            done = done and bool(solved.success)
                            brackets.append(dict(bounds=[trace[i-1].point[1-axis], trace[i+1].point[1-axis]],
                                                 calls=calls, complete=bool(solved.success), observations=observations))
                        except AttemptLimit:
                            brackets.append(dict(bounds=[trace[i-1].point[1-axis], trace[i+1].point[1-axis]],
                                                 calls=calls, complete=False, observations=observations))
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
                    options=dict(maxiter=180, ftol=1e-12, xtol=1e-10)), 250,
                start=start.point, start_candidate=start.candidate_id)

    def coarse(self):
        return self.search(BY_NAME['B00'])

    def _pool(self, result, domain, external):
        super()._pool(result, domain, external)
        if any(a.get('interrupted') is not None for a in result.attempts):
            result.complete = False
            result.unresolved_reason = 'mandatory_attempt_interrupted'

    def fine(self, result):
        self.partial_result = result
        if self.objective.budget.stop_reason or self.objective.landscape in self.objective.budget.arm_stops:
            result.complete = False
            return result
        return self.refine(result, audit_only=False)


def pre_controls(obj, family, replicate):
    points = interior(family, replicate, float(np.ptp(obj.simulator._mixer_eigenvalues)))
    for p in points:
        with obj.budget.trace() as work:
            check = native_gradient_check(obj, p, 'B00')
        yield seal_control(obj, check, work)
    for p in [*points, *product((0., 2*pi), (0., pi))]:
        yield direct_oracle(obj, p, 'B00')


def seal_control(obj, check, work):
    core = dict(check, work=work)
    control_id = obj.budget.record(dict(kind='control', control=core))
    return dict(core, control_id=control_id)


def direct_oracle(obj, point, domain='B00', *, candidate=None, winner_role=None):
    """Same approved direct-expm arithmetic, with linked admission and result."""
    with obj.budget.trace() as work:
        key = obj.key(*point)
        request = obj.budget.request(obj.landscape, domain, key=key, category='oracle', fresh=True)
        computation, exponentials = None, 0
        try:
            computation = obj.budget.charge(key, obj.landscape, domain, oracle=True, fresh=True, request_id=request)
            sim = obj.simulator
            obj.budget.counts['matrix_exponentials'] += 2
            exponentials = 2
            u, beta = point
            state = expm(-1j*beta*sim.mixer_hamiltonian)@expm(-1j*u*np.diag(sim.normalized_costs))@sim.initial_state
            obj.budget.outcome(computation, dict(gap=float(abs(state)**2@sim.normalized_costs),
                state_sha256=array_hash(state)), matrix_exponentials=exponentials)
            obj.budget.resolve(request, computation)
        except BaseException as error:
            if computation:
                obj.budget.outcome(computation, error=error, matrix_exponentials=exponentials)
            obj.budget.resolve(request, computation, error=error)
            raise
        native = obj.evaluate(*point, domain)
        probabilities = abs(state)**2
        errors = dict(probability=float(np.max(abs(probabilities-abs(native.state)**2))),
            gap=abs(float(probabilities@sim.normalized_costs)-native.gap),
            normalization=abs(float(np.sum(probabilities))-1.),
            native_normalization=abs(float(np.sum(abs(native.state)**2))-1.),
            native_state_norm=abs(float(np.linalg.norm(native.state))-1.))
        check = dict(kind='direct_expm', point=tuple(point), domain=domain, native=native.record(),
            native_computation_id=native.computation_id, oracle_computation_id=computation,
            status='pass' if max(errors.values()) <= 1e-10 else 'fail', errors=errors)
        if candidate is not None:
            check.update(candidate_id=candidate.candidate_id, winner_role=winner_role)
    obj.budget.check_time()
    return seal_control(obj, check, work)


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
        with obj.budget.trace() as work:
            a, b = obj.evaluate(*point), other.evaluate(*point)
        error = max(abs(a.gap-b.gap), float(np.max(abs(abs(a.state[mapping])**2-abs(b.state)**2))))
        yield seal_control(obj, dict(kind='cyclic_transport', domain='B00', point=point, error=error,
            native_computation_id=a.computation_id, transported_computation_id=b.computation_id,
            status='pass' if error <= 1e-10 else 'fail', transported_identity=other.identity), work)


def final_controls(obj, result):
    winners = [min((c for c in result.candidates if c.route == route), key=lambda c: (c.value, c.u, c.beta))
               for route in ('A', 'B')]
    winners.append(result.selected)
    for role, c in zip(('route_a', 'route_b', 'selected'), winners):
        with obj.budget.trace() as work:
            checked = obj.evaluate(*c.point, fresh=True)
        error = abs(checked.gap-c.value)
        check = dict(kind='candidate_reevaluation', domain='B00', point=c.point, error=error,
            candidate_id=c.candidate_id, winner_role=role, checked_computation_id=checked.computation_id,
            value=c.value, checked_value=checked.gap, status='pass' if error <= 1e-10 else 'fail')
        yield seal_control(obj, check, work)
        # The required repeated winners remain distinct controls even at one point.
        yield direct_oracle(obj, c.point, 'B00', candidate=c, winner_role=role)
