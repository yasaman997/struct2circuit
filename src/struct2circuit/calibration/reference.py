"""Empirical grid/local and independent DE references, never global certificates."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from itertools import product

import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

from .budget import BudgetExhausted
from .config import BY_NAME, PRIMARY, de_seed, grid_axes
from .config import numerical_settings
from .decisions import agreement, reference_status, refinement


class AttemptLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    u: float
    beta: float
    value: float
    route: str
    source: str
    success: bool | None = None
    seed: int | None = None
    objective_identity: str | None = None

    @property
    def point(self):
        return (self.u, self.beta)


@dataclass
class ReferenceResult:
    domain: str
    candidates: list[Candidate] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    route_a: float | None = None
    route_b: float | None = None
    pooled: float | None = None
    r: float | None = None
    f: float | None = None
    coarse: float | None = None
    faces_complete: bool = False
    complete: bool = False
    refined: bool = False
    full_bundle: bool = False
    unresolved_reason: str | None = None
    flat_faces: list[str] = field(default_factory=list)

    @property
    def selected(self):
        return min(self.candidates, key=lambda c: (c.value, c.u, c.beta)) if self.candidates else None

    def record(self):
        return asdict(self)


class ReferenceSearch:
    def __init__(self, objective, n_beta, identity, *, candidate_callback=None):
        self.objective, self.n_beta = objective, n_beta
        self.instance_id, self.graph_id, self.init_id = identity
        self.trace_cache = {}
        self.callback = candidate_callback

    def _candidate(self, point, domain, route, source, *, fresh=False, seed=None, success=None):
        if not domain.contains(point):
            raise ArithmeticError("Reference candidate outside its declared box")
        entry = self.objective.evaluate(*point, domain.name, fresh=fresh)
        if entry.gap is None:
            raise ValueError("No normalized objective for constant/unresolved costs")
        c = Candidate(*map(float, point), entry.gap, route, source, success, seed, self.objective.identity)
        if self.callback:
            self.callback(domain.name, c, entry)
        return c

    def _attempt(self, result, domain, route, source, runner, cap, seed=None):
        calls, observed = 0, []
        interrupted = None
        start_actual = self.objective.budget.counts['actual_evaluations']
        start_cpu = self.objective.budget.cpu_clock()

        def fun(point):
            nonlocal calls
            if calls >= cap:
                raise AttemptLimit("solver_objective_call_cap")
            calls += 1
            c = self._candidate(point, domain, route, source, seed=seed)
            observed.append(c)
            return c.value

        try:
            solved = runner(fun)
            if domain.contains(solved.x if np.ndim(solved.x) else (0, 0)):
                # Local/global candidates are freshly checked before being selected.
                c = self._candidate(solved.x, domain, route, source, fresh=True,
                                    seed=seed, success=bool(solved.success))
                observed.append(c)
            else:
                raise ArithmeticError("Solver returned an out-of-domain candidate")
            success, message = bool(solved.success), str(solved.message)
        except (AttemptLimit, BudgetExhausted) as error:
            success, message, interrupted = False, str(error), type(error).__name__
        result.candidates.extend(observed)
        result.attempts.append(dict(route=route, source=source, seed=seed, objective_calls=calls,
                                    actual_evaluations=self.objective.budget.counts['actual_evaluations']-start_actual,
                                    cpu_seconds=self.objective.budget.cpu_clock()-start_cpu,
                                    bounds=domain.bounds, objective_call_cap=cap,
                                    settings=numerical_settings()['de' if source.startswith('de_') and not source.startswith('de_polish') else 'powell' if source == 'powell' else 'local'],
                                    success=success, message=message, interrupted=interrupted))
        if isinstance(interrupted, str) and interrupted == 'BudgetExhausted':
            raise BudgetExhausted(message)
        return min(observed, key=lambda c: (c.value, c.u, c.beta)) if observed else None

    def _local(self, result, domain, start, route='A', source='lbfgsb', seed=None):
        candidate = self._attempt(result, domain, route, source,
            lambda fun: minimize(fun, np.asarray(start), method='L-BFGS-B',
                jac=lambda x: self.objective.gradient(x, domain.name), bounds=domain.bounds,
                options=dict(maxiter=120, ftol=1e-13, gtol=1e-9)), 250, seed)
        result.attempts[-1]['start'] = tuple(map(float, start))
        return candidate

    def _global(self, result, domain, replicate):
        seed = de_seed(self.instance_id, self.graph_id, self.init_id, domain.box_id, replicate)
        candidate = self._attempt(result, domain, 'B', f'de_{replicate}',
            lambda fun: differential_evolution(fun, domain.bounds, seed=seed, strategy='best1bin',
                popsize=8, maxiter=63 if replicate == 0 else 126, tol=1e-8, atol=1e-10,
                mutation=(.5, 1), recombination=.7, init='latinhypercube',
                updating='immediate', workers=1, polish=False), 1024 if replicate == 0 else 2032, seed)
        if candidate:
            self._local(result, domain, candidate.point, route='B', source=f'de_polish_{replicate}', seed=seed)

    def _starts(self, grid, production, axes, count):
        chosen = []
        ordered = sorted(grid, key=lambda c: (c.value, c.u, c.beta))
        if ordered:
            chosen.append(ordered[0])
        if production and all(production.point != c.point for c in chosen):
            chosen.append(production)
        def grid_index(candidate):
            return tuple(int(np.searchsorted(axes[i], candidate.point[i])) for i in (0, 1))
        for c in ordered:
            if len(chosen) >= count:
                break
            if c.point in {d.point for d in chosen}: continue
            index = grid_index(c)
            if all(max(abs(index[i]-grid_index(d)[i]) for i in (0, 1)) >= 2
                   for d in chosen if d is not production):
                chosen.append(c)
        return chosen[:count]

    def _faces(self, result, domain, axes, level):
        """Shared primary full traces; split scalar brackets at internal box cuts."""
        result.faces_complete = False
        base = BY_NAME['B11'] if domain.name in {b.name for b in PRIMARY} else domain
        base_axes = grid_axes(base, self.n_beta, level)
        traces = [(0, float(x)) for x in (0, 2*np.pi, 4*np.pi)] + [(1, float(x)) for x in (0, np.pi, 2*np.pi)] if base.name == 'B11' else [
            (axis, endpoint) for axis in (0, 1) for endpoint in base.bounds[axis]]
        complete = True
        for axis, fixed in traces:
            trace_key = (base.name, level, axis, fixed)
            if trace_key in self.trace_cache:
                candidates, done, flat = self.trace_cache[trace_key]
            else:
                variable = 1-axis
                nodes = base_axes[variable]
                trace = []
                for x in nodes:
                    c = self._candidate((fixed, x) if axis == 0 else (x, fixed), base, 'A', 'face_grid')
                    trace.append(c)
                    if domain.contains(c.point): result.candidates.append(c)
                candidates = list(trace)
                flat = axis == 1 and fixed == 0
                if axis == 0 and fixed == 0:
                    sim = self.objective.simulator
                    state = sim.initial_state
                    eigenvalue = np.vdot(state, sim.mixer_hamiltonian@state).real
                    residual = np.linalg.norm(sim.mixer_hamiltonian@state-eigenvalue*state)
                    flat = 2*max(abs(x) for x in base.beta)*residual <= 1e-10 and np.ptp([c.value for c in trace]) <= 1e-10
                done = True
                start_actual = self.objective.budget.counts['actual_evaluations']
                if not flat:
                    for i in range(1, len(trace)-1):
                        if trace[i].value >= min(trace[i-1].value, trace[i+1].value):
                            continue
                        lo, hi = float(nodes[i-1]), float(nodes[i+1])
                        # Primary internal box cuts are grid nodes, never bracketed across.
                        cuts = (2*np.pi,) if variable == 0 else (np.pi,)
                        edges = [lo] + [cut for cut in cuts if base.name == 'B11' and lo < cut < hi] + [hi]
                        capped = False
                        for left, right in zip(edges, edges[1:]):
                            calls = 0
                            def fun(x):
                                nonlocal calls
                                if calls >= 60 or self.objective.budget.counts['actual_evaluations']-start_actual >= 150:
                                    raise AttemptLimit('boundary_trace_cap')
                                calls += 1
                                point = (fixed, x) if axis == 0 else (x, fixed)
                                c = self._candidate(point, base, 'A', 'face_scalar')
                                candidates.append(c)
                                if domain.contains(c.point): result.candidates.append(c)
                                return c.value
                            try:
                                solve = minimize_scalar(fun, bounds=(left, right), method='bounded', options=dict(xatol=1e-10, maxiter=60))
                                done = done and bool(solve.success)
                            except AttemptLimit:
                                done, capped = False, True
                                break
                            except BudgetExhausted:
                                self.trace_cache[trace_key] = (candidates, False, flat)
                                result.complete = False
                                raise
                        if capped:
                            break
                self.trace_cache[trace_key] = (candidates, done, flat)
            result.candidates.extend(c for c in candidates if domain.contains(c.point))
            # Only intersecting required box faces contribute completeness.
            if any(fixed == endpoint for endpoint in domain.bounds[axis]):
                complete = complete and done
                if flat: result.flat_faces.append(f'{axis}:{fixed}')
        result.faces_complete = complete

    def _pool(self, result, domain, external):
        imported = {(c.point, c.source): c.value for c in result.candidates if c.route == 'pool'}
        for c in external:
            if c.objective_identity != self.objective.identity:
                raise ValueError('Nested candidate belongs to a different objective contract')
            if domain.contains(c.point):
                source = f"contained:{c.route}:{c.source}"
                if (c.point, source) in imported:
                    if abs(imported[c.point, source]-c.value) > 1e-10:
                        raise ArithmeticError('Repeated nested candidate has a different value')
                    continue
                evaluated = self.objective.evaluate(*c.point, domain.name)
                if evaluated.gap is None or abs(evaluated.gap-c.value) > 1e-10:
                    raise ArithmeticError('Nested candidate value disagrees with native evaluation')
                result.candidates.append(replace(c, route="pool", source=source))
                imported[c.point, source] = c.value
        for route, attr in (('A', 'route_a'), ('B', 'route_b')):
            values = [c.value for c in result.candidates if c.route == route]
            setattr(result, attr, min(values) if values else None)
        result.pooled = result.selected.value if result.selected else None
        result.r = agreement(result.route_a, result.route_b, result.pooled)
        if result.refined:
            result.f = refinement(result.coarse, result.pooled)
        result.complete = result.faces_complete and reference_status(result.r) == 'pass'
        if result.refined:
            result.complete = result.complete and reference_status(result.f) == 'pass'
        result.unresolved_reason = None if result.complete else 'reference_or_boundary_unresolved'

    def search(self, domain, *, production=None, contained=()):
        result = ReferenceResult(domain.name)
        if self.objective.simulator.cost_status != 'nonconstant':
            result.unresolved_reason = 'constant_or_unresolved'
            return result
        try:
            axes = grid_axes(domain, self.n_beta)
            grid = []
            for point in product(*axes):
                c = self._candidate(point, domain, 'A', 'grid')
                grid.append(c)
                result.candidates.append(c)
            for start in self._starts(grid, production, axes, 4):
                self._local(result, domain, start.point)
            self._global(result, domain, 0)
            self._faces(result, domain, axes, 0)
            self._pool(result, domain, tuple(contained)+(tuple([production]) if production else ()))
        except BudgetExhausted as error:
            result.unresolved_reason = str(error)
            result.complete = False
        return result

    def refine(self, result, *, production=None, contained=(), audit_only=False):
        domain = BY_NAME.get(result.domain)
        if domain is None:
            raise ValueError('Use refine_domain for an optional Bv domain')
        return self.refine_domain(result, domain, production=production, contained=contained, audit_only=audit_only)

    def refine_domain(self, result, domain, *, production=None, contained=(), audit_only=False):
        if result.refined:
            raise ValueError('Only one finer-grid level is approved per box')
        coarse = result.pooled
        result.refined = True
        result.coarse = coarse
        try:
            axes = grid_axes(domain, self.n_beta, 1)
            grid = []
            for point in product(*axes):
                c = self._candidate(point, domain, 'A', 'fine_grid')
                grid.append(c)
                result.candidates.append(c)
            for start in self._starts(grid, production, axes, 4 if audit_only else 8):
                self._local(result, domain, start.point, source='fine_lbfgsb')
            self._faces(result, domain, axes, 1)
            if not audit_only:
                self._bundle(result, domain)
            self._pool(result, domain, contained)
            result.f = refinement(coarse, result.pooled)
            self._pool(result, domain, ())
            # Remaining triggered work is a later scheduling wave, so an audit
            # cannot consume reserve before another mandatory audit/sentinel.
        except BudgetExhausted as error:
            result.complete = False
            result.unresolved_reason = str(error)
        return result

    def complete_bundle(self, result, domain, *, production=None):
        if not result.refined:
            raise ValueError('Complete the one finer grid before the remaining bundle')
        done = {tuple(a['start']) for a in result.attempts if a['source'] == 'fine_lbfgsb' and 'start' in a}
        axes = grid_axes(domain, self.n_beta, 1)
        grid = [c for c in result.candidates if c.source == 'fine_grid']
        for start in self._starts(grid, production, axes, 8):
            if start.point not in done:
                self._local(result, domain, start.point, source='fine_lbfgsb')
        self._bundle(result, domain)
        self._pool(result, domain, ())
        return result

    def _bundle(self, result, domain):
        if result.full_bundle:
            raise ValueError('Second DE/Powell bundle already used')
        result.full_bundle = True
        self._global(result, domain, 1)
        distinct = {}
        for c in sorted(result.candidates, key=lambda c: (c.value, c.u, c.beta)):
            distinct.setdefault(c.point, c)
        for start in list(distinct.values())[:2]:
            # A shared injected winner must not become a new independent route.
            self._attempt(result, domain, start.route, 'powell',
                lambda fun, p=start.point: minimize(fun, np.asarray(p), method='Powell', bounds=domain.bounds,
                    options=dict(maxiter=180, ftol=1e-12, xtol=1e-10)), 250)
