"""Bounded phase workflow. Official generation requires explicit execution authorization."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations, product
from pathlib import Path
import json
import os

import numpy as np

from .. import problems
from ..simulator import FeasibleSubspaceQAOA
from .budget import Budget, BudgetExhausted
from .config import (AUDIT_GRAPHS, BASELINE, BY_NAME, FAMILIES, INITIALIZATIONS, PRIMARY,
                     PROTOCOL_HASH, SCOPE, GRAPH_LABELS, SENTINELS, beta_intervals, build_manifest, construct_graphs,
                     instance_specs, de_seed, verify_protocol, verify_public_seed_metadata)
from .decisions import (Check, DEFICIT, DISTORTION, GAIN, agreement, band, challenge_gains,
                        classify, contrast, deficit, distortion, escalation_reasons,
                        reference_status, reversal, ordering)
from .diagnostics import (direct_oracle, gradient_check, invariance_checks, reuse_initialization,
                          spectral_coordinates, spectral_diagnostics)
from .objective import Objective, array_hash, production_search
from .reference import Candidate, ReferenceSearch
from .storage import (OutputStore, ValidationGate, create_validation_lock, driver_identity,
                      provenance, preflight_validation_lock, ARTIFACTS, PAIR_FIELDS)


@dataclass(frozen=True)
class Case:
    instance_id: int
    family: int
    graph_id: int
    initialization_id: int
    canonical_graph_id: int
    simulator: FeasibleSubspaceQAOA
    n_beta: int
    audit: bool = False
    sentinel: bool = False
    split: str = 'synthetic'
    shuffle_permutation: tuple[int, ...] = ()

    @property
    def key(self): return (self.instance_id, self.graph_id, self.initialization_id)

    @property
    def canonical_key(self): return (self.instance_id, self.canonical_graph_id, self.initialization_id)


@dataclass
class CaseResult:
    case: Case
    objective: Objective
    search: ReferenceSearch
    references: dict = field(default_factory=dict)
    production: dict = field(default_factory=dict)
    controls: list = field(default_factory=list)
    error: str | None = None
    arithmetic_failure: bool = False
    initial_metrics: dict | None = None
    required_challenges: bool = False
    paired_trigger: bool = False
    ranking_boxes: set = field(default_factory=set)


def require_execution_authorization(path, phase):
    if phase not in ('development', 'validation', 'optional'):
        raise ValueError('Unknown calibration phase')
    if path is None:
        raise PermissionError('Official calibration execution needs separate persisted authorization')
    authorization = json.loads(Path(path).read_text())
    if (authorization.get('calibration_execution_authorized') is not True
        or authorization.get('protocol_sha256') != PROTOCOL_HASH
        or phase not in authorization.get('phases', ())):
        raise PermissionError('Execution authorization does not cover this protocol and phase')
    return authorization


def generate_cases(specs, budget, *, authorization, phase, validation_gate=None, policy=None, reserve_activation=False):
    """Called only by an explicitly authorized later execution, never manifest building."""
    require_execution_authorization(authorization, phase)
    if phase == 'validation':
        if validation_gate is None:
            raise PermissionError('Validation requires its persisted lock before QUBO generation')
        validation_gate.begin_generation(policy)
    specs = tuple(specs)
    if any(s.split == 'reserve' for s in specs) and not reserve_activation:
        raise PermissionError('Reserved instances require the predeclared production-policy activation')
    allowed = {s.instance_id: s for s in instance_specs(True)}
    if any(s != allowed.get(s.instance_id) for s in specs):
        raise ValueError('Only exact approved instance specifications may enter an official phase')
    if any((s.split == 'validation') != (phase == 'validation') for s in specs):
        raise ValueError('Development/validation instance separation violated')
    cases = []
    for spec in specs:
        budget.check_time()
        if phase == 'validation': validation_gate.authorize(policy)
        generator, parameters = FAMILIES[spec.family]
        problem = getattr(problems, generator)(n=spec.n, k=spec.k, seed=spec.seed, **parameters)
        graphs, canonical, permutation = construct_graphs(problem, spec.instance_id)
        bases = {}
        for g in sorted(set(canonical)):
            bases[g] = FeasibleSubspaceQAOA(problem, graphs[g])
            budget.counts['eigensolver_calls'] += 1
            budget.check_time()
        n_beta = beta_intervals(max(np.ptp(s._mixer_eigenvalues) for s in bases.values()))
        for graph_id in range(8):
            for init_id, init in enumerate(INITIALIZATIONS):
                sim = reuse_initialization(bases[canonical[graph_id]], init, budget)
                cases.append(Case(spec.instance_id, spec.family, graph_id, init_id, canonical[graph_id], sim,
                                  n_beta, graph_id == AUDIT_GRAPHS[spec.family],
                                  (graph_id, init_id) == SENTINELS[spec.family], spec.split, tuple(permutation)))
    return tuple(cases)


def _points(case):
    seed = 820000000+10000*case.instance_id+100*case.canonical_graph_id+10*case.initialization_id
    rng = np.random.default_rng(seed)
    width = float(np.ptp(case.simulator._mixer_eigenvalues))
    margins = np.array((1e-5, 1e-5/max(1., width)))
    # Independent fixed-seed uniform interior samples; implementation rule is
    # recorded before execution and remains unchanged on validation.
    interior = rng.uniform(margins, np.array((4*np.pi, 2*np.pi))-margins, size=(8, 2))
    corners = tuple(product((0., 2*np.pi, 4*np.pi), (0., np.pi, 2*np.pi)))
    control = tuple(product(np.linspace(0, 2*np.pi, 5), np.linspace(0, np.pi, 5)))
    return interior, corners, control


class PhaseDriver:
    """Same reference rules for development and validation; one phase, no implicit holdout."""
    def __init__(self, cases, budget, store, *, expected_keys, policies=(9, 17),
                 broader_essential=False, validation_gate=None, synthetic=False):
        self.cases, self.budget, self.store = tuple(cases), budget, store
        self.expected_keys = tuple(tuple(k) for k in expected_keys)
        self.policies = tuple(policies)
        self.broader_essential = broader_essential
        self.gate, self.synthetic = validation_gate, synthetic
        self.locked_policy = self.gate.payload["policy"] if self.gate else None
        self.validation_policies = self.policies if self.gate else None
        self.results, self.aliases, self.pairs = {}, {}, []
        self.phase_failures = []
        self.pending_work = {}
        if len(self.expected_keys) != len(set(self.expected_keys)):
            raise ValueError('Duplicate expected landscape keys')
        if any(n not in (9, 17, 33) for n in policies) or not policies:
            raise ValueError('Only B00 P9/P17/P33 policies supported')
        if synthetic and any(c.instance_id in range(12) or c.simulator.problem.n > 4 for c in cases):
            raise ValueError('Synthetic workflow may not use approved study instance IDs')
        if any(c.split == 'validation' for c in cases):
            if self.gate is None:
                raise PermissionError('Validation requires a persisted lock')
            if 33 in self.policies and not self.gate.payload['reserves_active']:
                raise ValueError('P33 diagnostic was not activated before validation')
            if self.gate.payload['policy'] not in self.policies:
                raise ValueError('Locked production policy absent from requested policies')
            if broader_essential != self.gate.payload['manifest']['settings']['broader_adequacy_essential']:
                raise ValueError('Validation acceptance target differs from its lock')
        if not synthetic and any(c.split == 'synthetic' for c in cases):
            raise ValueError('Synthetic objects must be explicitly labeled')

    def _emit_candidate(self, key, domain, candidate, evaluation):
        if candidate.source not in ('grid', 'fine_grid', 'face_grid'):
            self.store.row('candidates.jsonl', dict(landscape=key, domain=domain,
                candidate=asdict(candidate), evaluation=evaluation.record(),
                accounting=dict(self.budget.counts), cpu_seconds=self.budget.prior_cpu+self.budget.cpu_clock()-self.budget.cpu_start))

    def _routine(self, case):
        key = ':'.join(map(str, case.canonical_key))
        obj = Objective(case.simulator, self.budget, key,
                        account='shared' if case.split == 'reserve' else 'essential')
        search = ReferenceSearch(obj, case.n_beta, case.canonical_key,
                                 candidate_callback=lambda b, c, e: self._emit_candidate(case.key, b, c, e))
        result = CaseResult(case, obj, search, required_challenges=case.sentinel)
        self.results[case.canonical_key] = result
        if case.simulator.cost_status != 'nonconstant':
            result.error = 'constant_or_unresolved'
            return
        try:
            initial = obj.evaluate(0., 0.)
            result.initial_metrics = initial.record() | dict(initial_optimum_probability=float(np.sum(abs(initial.state[case.simulator.optimal_mask])**2)))
            interior, corners, control = _points(case)
            for point in interior:
                self._run_control(result, direct_oracle, obj, point, 'B11')
                self._record_control(result, gradient_check(obj, point, 'B11'))
            if any(c['status'] == 'fail' for c in result.controls):
                raise ArithmeticError('Pre-search pointwise oracle or analytic derivative mismatch')
            for n in self.policies:
                optimized, value = production_search(obj, n)
                result.production[n] = Candidate(value.u, value.beta, value.gap, 'production', f'P{n}', objective_identity=obj.identity)
                self.store.row('candidates.jsonl', dict(landscape=case.key, domain='B00', policy=n,
                    evaluation=value.record(), optimizer=asdict(optimized), accounting=dict(self.budget.counts)))
            production = min(result.production.values(), key=lambda c: c.value)
            for domain in PRIMARY:
                contained = tuple(c for ref in result.references.values() for c in ref.candidates)
                result.references[domain.name] = search.search(domain, production=production, contained=contained)
                if self.budget.stop_reason: raise BudgetExhausted(self.budget.stop_reason)
            for point in corners:
                self._run_control(result, direct_oracle, obj, point, 'B11')
            for point in control:
                self._run_control(result, spectral_coordinates, obj, point)
            if case.split in ('development', 'synthetic'):
                invariance_checks(obj, control, record_callback=lambda check: self._record_control(result, check))
            self._selected_oracles(result)
        except BudgetExhausted as error:
            result.error = str(error)
        except (ArithmeticError, ValueError) as error:
            result.error = str(error)
            result.arithmetic_failure = True

    def _record_control(self, result, check):
        check = dict(check, tolerance=1e-6 if check.get('kind') == 'gradient' else 1e-10,
                     objective_identity=result.objective.identity)
        result.controls.append(check)
        if check['status'] == 'fail':
            result.arithmetic_failure = True
            self.store.row('candidates.jsonl', dict(landscape=result.case.key,
                kind='numerical_control', control=check, accounting=dict(self.budget.counts)))

    def _run_control(self, result, diagnostic, *args):
        # Record completed comparisons before their terminal budget check.
        recorded = False
        def record(check):
            nonlocal recorded
            self._record_control(result, check)
            recorded = True
        check = diagnostic(*args, record_callback=record)
        if not recorded:
            self._record_control(result, check)

    def _selected_oracles(self, result):
        for domain, ref in result.references.items():
            if ref.selected:
                native = result.objective.evaluate(*ref.selected.point, domain, fresh=True)
                error = abs(native.gap-ref.selected.value)
                self._record_control(result, dict(status='pass' if error <= 1e-10 else 'fail',
                    kind='candidate_reevaluation', point=ref.selected.point, domain=domain, error=error))
                self._run_control(result, direct_oracle, result.objective, ref.selected.point, domain)
        for candidate in result.production.values():
            self._run_control(result, direct_oracle, result.objective, candidate.point)

    def _reference_checks(self, result):
        reasons = {}
        r00 = result.references.get('B00')
        if not r00: return reasons
        for name, ref in result.references.items():
            base = result.references.get('B11') if name == 'Bsign' else r00
            gains = () if name == 'B00' else (deficit(base.pooled if base else None, ref.pooled),)
            near = False
            if ref.selected:
                domain = BY_NAME[name]
                lower, upper = domain.face_distances(ref.selected.point)
                for axis in (0, 1):
                    for fixed, distance in ((domain.bounds[axis][0], lower[axis]), (domain.bounds[axis][1], upper[axis])):
                        if distance <= .01 and f'{axis}:{fixed}' not in ref.flat_faces:
                            near = True
            e = max((deficit(c.value, r00.pooled) for c in result.production.values()), default=None) if name == 'B00' and r00.pooled is not None else None
            reasons[name] = escalation_reasons(r=ref.r, faces_complete=ref.faces_complete, e=e,
                                              gains=gains, near_nonflat=near, rank_unresolved=name in result.ranking_boxes,
                                              audit=result.case.audit and name in {b.name for b in PRIMARY})
            if ref.refined and reference_status(ref.f) != 'pass':
                reasons[name] += ('reference_refinement',)
        return reasons

    def _paired_rows(self):
        self.pairs = []
        for result in self.results.values():
            result.paired_trigger = False
            result.ranking_boxes.clear()
        by_stratum = {}
        for case in self.cases:
            result = self.results.get(case.canonical_key)
            if result: by_stratum.setdefault((case.instance_id, case.initialization_id), []).append((case, result))
        for (instance_id, init), entries in by_stratum.items():
            for (left, a), (right, b) in combinations(sorted(entries, key=lambda x: x[0].graph_id), 2):
                ra, rb = a.references.get('B00'), b.references.get('B00')
                for n in self.policies:
                    pa, pb = a.production.get(n), b.production.get(n)
                    k = distortion(pa.value if pa else None, pb.value if pb else None,
                                   ra.pooled if ra and ra.complete else None, rb.pooled if rb and rb.complete else None)
                    self.pairs.append(dict(instance_id=instance_id, initialization_id=init, i=left.graph_id, j=right.graph_id,
                                           policy=n, production_contrast=contrast(pa.value, pb.value) if pa and pb else None,
                                           reference_contrast=contrast(ra.pooled, rb.pooled) if ra and rb and ra.complete and rb.complete else None,
                                           K=k, status=band(k, DISTORTION),
                                           production_order=ordering(contrast(pa.value, pb.value)) if pa and pb else 'unresolved',
                                           reference_order=ordering(contrast(ra.pooled, rb.pooled)) if ra and rb and ra.complete and rb.complete else 'unresolved'))
                    if band(k, DISTORTION) in ('fail', 'ambiguous'):
                        for result in (a, b): result.paired_trigger = True
                domains = PRIMARY+tuple(BY_NAME[name] for name in ('B22', 'Bsign') if a.required_challenges and b.required_challenges)
                for first, second in combinations(domains, 2):
                    refs = [r.references.get(domain.name) for r in (a, b) for domain in (first, second)]
                    first_contrast = second_contrast = None
                    change = 'unresolved'
                    if all(ref and ref.complete for ref in refs):
                        first_contrast = contrast(refs[0].pooled, refs[2].pooled)
                        second_contrast = contrast(refs[1].pooled, refs[3].pooled)
                        change = reversal(first_contrast, second_contrast)
                        if change in ('reversal', 'ambiguous') and first in PRIMARY and second in PRIMARY:
                            a.required_challenges = b.required_challenges = True
                            if change == 'ambiguous':
                                for result in (a, b): result.ranking_boxes.update((first.name, second.name))
                    self.pairs.append(dict(instance_id=instance_id, initialization_id=init, i=left.graph_id, j=right.graph_id,
                                           policy=None, production_contrast=None, reference_contrast=None, K=None,
                                           status=change, domains=f'{first.name}/{second.name}',
                                           first_contrast=first_contrast, second_contrast=second_contrast,
                                           first_order=ordering(first_contrast), second_order=ordering(second_contrast)))

    def _scheduled_work(self):
        """Mandatory audit/sentinel wave precedes any triggered reserve work."""
        for result in self.results.values():
            if result.error or not result.references: continue
            try:
                production = min(result.production.values(), key=lambda c: c.value)
                if result.case.audit:
                    for domain in PRIMARY:
                        ref = result.references[domain.name]
                        if not ref.refined:
                            result.search.refine(ref, production=production, audit_only=True)
                if result.case.sentinel:
                    r11 = result.references['B11']
                    for name in ('B22', 'Bsign'):
                        if name not in result.references:
                            result.references[name] = result.search.search(BY_NAME[name], contained=tuple(r11.candidates))
            except BudgetExhausted as error:
                result.error = str(error)
            except (ArithmeticError, ValueError) as error:
                result.error, result.arithmetic_failure = str(error), True

    def _pool_references(self, result):
        # Narrow-to-wide order propagates improvements without creating routes.
        for domain in PRIMARY+tuple(BY_NAME[n] for n in ('B22', 'Bsign')):
            if domain.name not in result.references: continue
            contained = tuple(c for name, ref in result.references.items()
                              if name != domain.name and domain.includes(BY_NAME[name])
                              for c in ref.candidates)
            result.search._pool(result.references[domain.name], domain, contained)
        r00, r11 = result.references['B00'], result.references['B11']
        upper = BY_NAME['B11'].face_distances(r11.selected.point)[1] if r11.selected else (1, 1)
        result.required_challenges |= any(d <= .01 for d in upper) or band(deficit(r00.pooled, r11.pooled), GAIN) in ('fail', 'ambiguous')

    def _pending_tasks(self, result):
        if result.error or not result.references: return ()
        reasons = self._reference_checks(result)
        if result.paired_trigger:
            reasons['B00'] = reasons.get('B00', ()) + ('paired_distortion',)
        tasks = []
        for name in tuple(b.name for b in PRIMARY)+('B22', 'Bsign'):
            ref = result.references.get(name)
            if ref is None:
                if name in ('B22', 'Bsign') and result.required_challenges:
                    tasks.append((name, 'search', ()))
                continue
            reason = reasons.get(name, ())
            if reason and not ref.refined:
                tasks.append((name, 'refine', reason))
            elif reason and reason != ('scheduled_audit',) and not ref.full_bundle:
                tasks.append((name, 'bundle', reason))
        return tuple(tasks)

    def _settle(self):
        # Each box can be created/refined/bundled only once: at most 14 actions
        # per landscape (two challenge creations and two actions per six boxes).
        ordered = sorted(self.results.items(), key=lambda item: (item[1].case.initialization_id,
                         item[1].case.graph_id, item[1].case.instance_id))
        for wave in range(1+14*len(ordered)):
            self._paired_rows()
            progressed = False
            for key, result in ordered:
                if result.error or not result.references: continue
                try:
                    self._pool_references(result)
                    tasks = self._pending_tasks(result)
                    self.pending_work[key] = tasks
                    for name, action, reason in tasks:
                        self.budget.check_time()
                        previous = result.objective.account
                        light = reason == ('scheduled_audit',)
                        result.objective.account = ('essential' if result.case.split != 'reserve' and
                            (light or action == 'search' and result.case.sentinel) else 'shared')
                        try:
                            ref = result.references.get(name)
                            before = (ref is not None, bool(ref and ref.refined), bool(ref and ref.full_bundle))
                            domain = BY_NAME[name]
                            contained = tuple(c for other, ref in result.references.items()
                                if other != name and domain.includes(BY_NAME[other]) for c in ref.candidates)
                            production = min(result.production.values(), key=lambda c: c.value)
                            if action == 'search':
                                result.references[name] = result.search.search(domain, contained=contained)
                            elif action == 'refine':
                                result.search.refine(result.references[name], production=production,
                                                     contained=contained, audit_only=light)
                            else:
                                result.search.complete_bundle(result.references[name], domain, production=production)
                            ref = result.references[name]
                            after = (True, ref.refined, ref.full_bundle)
                            progressed |= before != after
                        finally:
                            result.objective.account = previous
                        if self.budget.stop_reason: raise BudgetExhausted(self.budget.stop_reason)
                    self._pool_references(result)
                    if wave == 0 or tasks:
                        self._selected_oracles(result)
                except BudgetExhausted as error:
                    result.error = str(error)
                except (ArithmeticError, ValueError) as error:
                    result.error, result.arithmetic_failure = str(error), True
            self._paired_rows()
            self.pending_work = {key: self._pending_tasks(result) for key, result in ordered}
            if self.budget.stop_reason or not any(self.pending_work.values()):
                return
            if not progressed:
                break
        # Pending work can never pass, even if a backend did not advance flags.
        # This bound protects against malformed callbacks as well as regressions.

    def checks(self, policy):
        checks, expected = [], []
        for key in self.expected_keys:
            case = next((c for c in self.cases if c.key == key), None)
            result = self.results.get(case.canonical_key) if case else None
            prefix = ':'.join(map(str, key))
            def add(name, status, essential=True):
                full = prefix+':'+name
                expected.append(full)
                checks.append(Check(full, status, essential))
            if result is None:
                add('landscape', 'missing')
                continue
            add('arithmetic', 'fail' if result.arithmetic_failure or any(c['status'] == 'fail' for c in result.controls) else 'unresolved' if result.error else 'pass')
            add('trigger_work', 'unresolved' if self.pending_work.get(case.canonical_key) else 'pass')
            for name in tuple(b.name for b in PRIMARY)+(('B22', 'Bsign') if result.required_challenges else ()):
                ref = result.references.get(name)
                add('reference:'+name, 'pass' if ref and ref.complete else 'unresolved')
            reference = result.references.get('B00')
            production = result.production.get(policy)
            add('E', band(deficit(production.value if production else None,
                                  reference.pooled if reference and reference.complete else None), DEFICIT))
            r00 = reference.pooled if reference and reference.complete else None
            for name in ('B10', 'B01', 'B11'):
                ref = result.references.get(name)
                add('gain:'+name, band(deficit(r00, ref.pooled if ref and ref.complete else None), GAIN), self.broader_essential)
            if result.required_challenges:
                r11, r22, rs = (result.references.get(n) for n in ('B11', 'B22', 'Bsign'))
                gains = challenge_gains(r00, r11.pooled if r11 and r11.complete else None,
                                        r22.pooled if r22 and r22.complete else None, rs.pooled if rs and rs.complete else None)
                for name in ('direct', 'step', 'signed'):
                    add('challenge:'+name, band(gains[name], GAIN), self.broader_essential)
            sensitivity = [c['selection_status'] for c in result.controls if 'selection_status' in c]
            add('label_sensitivity', 'ambiguous' if 'ambiguous' in sensitivity else 'fail' if 'fail' in sensitivity else 'pass', False)
        strata = {}
        for instance_id, graph_id, init_id in self.expected_keys:
            strata.setdefault((instance_id, init_id), []).append(graph_id)
        for (instance_id, init_id), graphs in strata.items():
            for i, j in combinations(sorted(graphs), 2):
                expected.append(f'pair:{instance_id}:{init_id}:{i}:{j}')
        for pair in self.pairs:
            if pair['policy'] == policy:
                name = f"pair:{pair['instance_id']}:{pair['initialization_id']}:{pair['i']}:{pair['j']}"
                checks.append(Check(name, pair['status']))
            elif pair['policy'] is None and pair['status'] == 'reversal':
                checks.append(Check(f"rank:{pair['instance_id']}:{pair['initialization_id']}:{pair['i']}:{pair['j']}:{pair['domains']}", 'fail', self.broader_essential))
        expected.append('resource_completion')
        checks.append(Check('resource_completion', 'unresolved' if self.budget.stop_reason and not self.budget.mandatory_complete else 'pass'))
        for i, message in enumerate(self.phase_failures):
            checks.append(Check(f'phase_failure:{i}:{message}', 'fail'))
        return checks, expected

    def status(self, policy):
        if self.gate: self.gate.authorize(policy)
        checks, expected = self.checks(policy)
        return classify(checks, expected, optional_complete=False)

    def run(self):
        if self.gate and self.policies != self.validation_policies:
            raise ValueError('Validation policy configuration was altered')
        for case in sorted(self.cases, key=lambda c: (c.initialization_id, c.graph_id, c.instance_id)):
            self.aliases[case.key] = case.canonical_key
            if case.canonical_key in self.results:
                result = self.results[case.canonical_key]
                result.required_challenges |= case.sentinel
                # A duplicate audit label still schedules the canonical calculation.
                if case.audit or case.sentinel:
                    from dataclasses import replace
                    result.case = replace(result.case, audit=result.case.audit or case.audit, sentinel=result.case.sentinel or case.sentinel)
                continue
            if self.gate: self.gate.authorize(self.gate.payload['policy'])
            if self.budget.stop_reason: break
            self._routine(case)
        self._scheduled_work()
        self._settle()
        return self

    def add_policy(self, policy):
        if self.gate: raise ValueError('Cannot retune policies after validation begins')
        if policy not in (9, 17, 33): raise ValueError('Unknown policy')
        if policy in self.policies: return
        self.policies += (policy,)
        for result in self.results.values():
            if result.error: continue
            try:
                old = result.objective.account
                result.objective.account = 'shared'
                _, value = production_search(result.objective, policy)
                result.objective.account = old
                c = Candidate(value.u, value.beta, value.gap, 'production', f'P{policy}', objective_identity=result.objective.identity)
                result.production[policy] = c
                result.search._pool(result.references['B00'], BY_NAME['B00'], (c,))
            except BudgetExhausted as error:
                result.error = str(error)
        self._settle()

    def write_results(self, policy, *, final_status=None):
        if final_status not in (None, 'FAILED', 'INCONCLUSIVE'):
            raise ValueError('A reporting override may only preserve failure or incompleteness')
        if self.gate: self.gate.authorize(policy)
        for case in self.cases:
            result = self.results.get(case.canonical_key)
            sim = case.simulator
            self.store.row('graphs.jsonl', dict(landscape=case.key, canonical=case.canonical_key, edges=sim.mixer.edges,
                           graph_label=GRAPH_LABELS[case.graph_id], graph_id=case.graph_id,
                           edge_hash=array_hash(np.asarray(sim.mixer.edges)), m=len(sim.mixer.edges),
                           initialization=sim.initialization, spectrum=spectral_diagnostics(sim),
                           shuffle_permutation=case.shuffle_permutation, random_seed=520000000+case.instance_id,
                           shuffle_seed=620000000+case.instance_id, initial_metrics=result.initial_metrics if result else None,
                           requested_DE_seeds=[de_seed(*case.key, b, r) for b in range(7) for r in (0, 1)],
                           canonical_DE_seeds=[de_seed(*case.canonical_key, b, r) for b in range(7) for r in (0, 1)],
                           initial_state_sha256=array_hash(sim.initial_state)))
            row = dict(landscape=case.key, canonical=case.canonical_key, status='missing' if result is None else 'observed',
                       normalized_gap=None, required_challenges=bool(result and result.required_challenges),
                       pending_trigger_work=self.pending_work.get(case.canonical_key, ()))
            if result:
                row.update(error=result.error, arithmetic_failure=result.arithmetic_failure,
                           cost_status=sim.cost_status, feasible_span=sim.feasible_span,
                           family=case.family, split=case.split, n=sim.problem.n, k=sim.problem.k, d=len(sim.basis),
                           production={str(n): asdict(c) for n, c in result.production.items()},
                           references={name: dict(pooled=r.pooled, route_a=r.route_a, route_b=r.route_b,
                               coarse=r.coarse, fine=r.pooled if r.refined else None, R=r.r, F=r.f, complete=r.complete,
                               selected=asdict(r.selected) if r.selected else None, attempts=r.attempts,
                               face_distances=BY_NAME[name].face_distances(r.selected.point) if r.selected else None,
                               boundary_hit=any(d <= 1e-8 for distances in BY_NAME[name].face_distances(r.selected.point) for d in distances) if r.selected else None,
                               near_boundary=any(d <= .01 for distances in BY_NAME[name].face_distances(r.selected.point) for d in distances) if r.selected else None,
                               flat_faces=r.flat_faces, faces_complete=r.faces_complete, refined=r.refined, unresolved_reason=r.unresolved_reason)
                                       for name, r in result.references.items()}, controls=result.controls)
                refs = {name: r.pooled if r.complete else None for name, r in result.references.items()}
                r00 = refs.get('B00')
                row.update(E={str(n): deficit(c.value, r00) for n, c in result.production.items()},
                           gains={name: deficit(r00, refs.get(name)) for name in ('B10', 'B01', 'B11')},
                           challenge_gains=challenge_gains(r00, refs.get('B11'), refs.get('B22'), refs.get('Bsign')))
            self.store.row('landscapes.jsonl', row)
        # Explicitly retain requested-but-unconstructed rows.
        observed = {c.key for c in self.cases}
        for key in self.expected_keys:
            if key not in observed: self.store.row('landscapes.jsonl', dict(landscape=key, status='missing', normalized_gap=None))
        for pair in self.pairs: self.store.pair({name: pair.get(name) for name in PAIR_FIELDS})
        computed = self.status(policy)
        status = 'FAILED' if computed == 'FAILED' else final_status or computed
        self.budget.mandatory_complete = status in ('ACCEPTED', 'ACCEPTED WITH EXPLICIT LIMITATIONS')
        checks, expected = self.checks(policy)
        observed = {c.name for c in checks}
        unfinished = [c.name for c in checks if c.status in ('unresolved', 'missing', 'ambiguous', 'omitted')]
        self.store.json('budget.json', self.budget.snapshot() | dict(incomplete_required_work=unfinished+sorted(set(expected)-observed)))
        groups = {}
        for check in checks:
            group = ('paired_B00_K' if check.name.startswith('pair:') else 'domain_ordering' if check.name.startswith('rank:')
                     else 'arithmetic' if ':arithmetic' in check.name else 'references' if ':reference:' in check.name
                     else 'B00_E' if check.name.endswith(':E') else 'positive_envelope' if ':gain:' in check.name or ':challenge:direct' in check.name or ':challenge:step' in check.name
                     else 'signed_sector' if ':challenge:signed' in check.name else 'label_sensitivity' if ':label_sensitivity' in check.name else 'coverage')
            counts = groups.setdefault(group, {})
            counts[check.status] = counts.get(check.status, 0)+1
        statistics, best_sets = self.descriptive_records(policy)
        qualification = status if self.gate or self.synthetic or status == 'FAILED' else 'INCONCLUSIVE'
        validation_completed = bool(self.gate) and self.budget.mandatory_complete
        self.store.decision(f'{qualification}\n\nPhase numerical status: {status}.\nPolicy: P{policy} on B00 only.\n'
            f'Scope: {SCOPE}. Validation completed: {validation_completed}. Synthetic check: {self.synthetic}.\n'
            f'Expected numerical checks: {len(expected)}.\n'
            f'Checks: {json.dumps([asdict(c) for c in checks])}\n'
            f'Separate claim counts: {json.dumps(groups)}\n'
            f'Descriptive E/K statistics (no average overrides failures): {json.dumps(statistics)}\n'
            f'Graphs within 2e-3 of each primary-box best (descriptive): {json.dumps(best_sets)}\n'
            'Expanded production policies: NOT TESTED.\n'
            'Common-spectral-time interpretation: OMITTED.\n'
            'No RMS superiority or global-optimum claim. Benchmark: DRAFT_UNFROZEN.')
        return status

    def descriptive_records(self, policy):
        values, best_sets = {}, []
        for case in self.cases:
            result = self.results.get(case.canonical_key)
            ref = result.references.get('B00') if result else None
            prod = result.production.get(policy) if result else None
            value = deficit(prod.value if prod else None, ref.pooled if ref and ref.complete else None)
            if value is not None:
                key = (case.family, case.initialization_id, case.graph_id, case.simulator.problem.n)
                values.setdefault(('E', key), []).append((case.key, value))
        for pair in self.pairs:
            if pair['policy'] != policy or pair['K'] is None: continue
            case = next(c for c in self.cases if c.key == (pair['instance_id'], pair['i'], pair['initialization_id']))
            key = (case.family, case.initialization_id, pair['i'], pair['j'], case.simulator.problem.n)
            values.setdefault(('K', key), []).append(((pair['instance_id'], pair['i'], pair['j']), pair['K']))
        statistics = []
        for (quantity, stratum), rows in sorted(values.items()):
            numbers = [v for _, v in rows]
            threshold = DEFICIT if quantity == 'E' else DISTORTION
            statistics.append(dict(quantity=quantity, stratum=stratum, count=len(rows), maximum=max(numbers),
                median=float(np.median(numbers)), percentile90=float(np.percentile(numbers, 90)),
                offending=[key for key, v in rows if band(v, threshold) in ('fail', 'ambiguous')]))
        for instance_id, init in sorted({(c.instance_id, c.initialization_id) for c in self.cases}):
            for domain in PRIMARY:
                rows = []
                for case in self.cases:
                    if (case.instance_id, case.initialization_id) != (instance_id, init): continue
                    result = self.results.get(case.canonical_key)
                    ref = result.references.get(domain.name) if result else None
                    if ref and ref.complete: rows.append((case.graph_id, ref.pooled))
                minimum = min((v for _, v in rows), default=None)
                best_sets.append(dict(instance_id=instance_id, initialization_id=init, domain=domain.name,
                    complete=len(rows) == sum((c.instance_id, c.initialization_id) == (instance_id, init) for c in self.cases),
                    graphs=[g for g, v in rows if v-minimum <= .002] if minimum is not None else None))
        return statistics, best_sets


def run_synthetic(cases, output, *, limits, expected_keys=None, policies=(9, 17)):
    budget = Budget(limits)
    keys = [c.key for c in cases] if expected_keys is None else expected_keys
    with OutputStore(output) as store:
        manifest = dict(kind='SYNTHETIC_IMPLEMENTATION_CHECK', expected_keys=keys, status='RUNNING',
                        policies=policies, provenance=provenance(), official_calibration_executed=False)
        store.json('calibration_manifest.json', manifest)
        for instance_id in sorted({c.instance_id for c in cases}):
            store.row('instances.jsonl', dict(instance_id=instance_id, split='synthetic', official=False))
        phase = PhaseDriver(cases, budget, store, expected_keys=keys, policies=policies, synthetic=True).run()
        status = phase.write_results(policies[-1])
        store.json('calibration_manifest.json', manifest | dict(status=status), update_owned=True)
    return status, budget.snapshot()


def run_official_phase(output, *, phase, authorization, protocol, lock=None, lock_output=None):
    """Future separately authorized execution. This function is not run in Stage 5A."""
    if phase not in ('development', 'validation'):
        raise ValueError('Official phase must be development or validation')
    verify_protocol(protocol)
    require_execution_authorization(authorization, phase)
    seed_metadata = verify_public_seed_metadata(Path(__file__).resolve().parents[3])
    gate = ValidationGate(lock) if phase == 'validation' and lock else None
    if phase == 'validation' and gate is None:
        raise PermissionError('A persisted validation lock is mandatory')
    if gate:
        preflight_validation_lock(gate.path.with_name(gate.path.name+'.validation-started'))
    budget = Budget.from_snapshot(gate.payload['budget']) if gate else Budget()
    reserves = gate.payload['reserves_active'] if gate else False
    policy = gate.payload['policy'] if gate else 17
    specs = tuple(s for s in instance_specs(reserves) if (s.split == 'validation') == (phase == 'validation'))
    expected = [(s.instance_id, g, init) for s in specs for g in range(8) for init in range(3)]
    # All output preflight occurs before ANY instance generation or simulation.
    if lock_output:
        preflight_validation_lock(lock_output)
        destinations = { (Path(output)/name).resolve() for name in ARTIFACTS } | {Path(output).resolve()}
        if Path(lock_output).resolve() in destinations:
            raise FileExistsError('Validation lock aliases a deterministic output destination')
    if any(os.environ.get(name) != '1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('Official execution requires one numerical thread; use the entry point in a new process')
    with OutputStore(output) as store:
        manifest = build_manifest(reserves) | dict(phase=phase, status='RUNNING', provenance=provenance(),
            authorized_execution=True, validation_lock=gate.payload if gate else None,
            authorization_record=require_execution_authorization(authorization, phase))
        manifest['seed_use_check'] = seed_metadata
        store.json('calibration_manifest.json', manifest)
        driver = None
        try:
            cases = generate_cases(specs, budget, authorization=authorization, phase=phase, validation_gate=gate, policy=policy)
            for spec in specs:
                sim = next(c.simulator for c in cases if c.instance_id == spec.instance_id)
                store.row('instances.jsonl', asdict(spec) | dict(status='generated', d=len(sim.basis), generator=FAMILIES[spec.family][0],
                    parameters=FAMILIES[spec.family][1], coefficients_sha256=array_hash(sim.problem.Q, sim.problem.c),
                    generator_source=driver_identity(), metadata=sim.problem.metadata))
            driver = PhaseDriver(cases, budget, store, expected_keys=expected, policies=(9, 17) if not gate else (policy,), validation_gate=gate).run()
            checks17, _ = driver.checks(17) if not gate else ((), ())
            numerical_failure = any(c.essential and c.status == 'fail' and (c.name.endswith(':E') or c.name.startswith('pair:')) for c in checks17)
            arithmetic_failure = any(r.arithmetic_failure for r in driver.results.values())
            if not gate and numerical_failure and not arithmetic_failure:
                driver.add_policy(33)
                reserves = True
                reserved_specs = tuple(s for s in instance_specs(True) if s.split == 'reserve')
                manifest.update(build_manifest(True))
                manifest['reserve_activation'] = 'P33 proposed after resolved P17 failure'
                specs += reserved_specs
                driver.expected_keys += tuple((s.instance_id, g, init) for s in reserved_specs for g in range(8) for init in range(3))
                reserved_cases = generate_cases(reserved_specs, budget, authorization=authorization, phase=phase, reserve_activation=True)
                driver.cases += reserved_cases
                for spec in reserved_specs:
                    sim = next(c.simulator for c in reserved_cases if c.instance_id == spec.instance_id)
                    store.row('instances.jsonl', asdict(spec) | dict(status='generated', activation='P33', d=len(sim.basis),
                        generator=FAMILIES[spec.family][0], parameters=FAMILIES[spec.family][1], generator_source=driver_identity(),
                        coefficients_sha256=array_hash(sim.problem.Q, sim.problem.c), metadata=sim.problem.metadata))
                driver.run()
                passing = [n for n in (17, 33, 9) if driver.status(n) in ('ACCEPTED', 'ACCEPTED WITH EXPLICIT LIMITATIONS')]
                if passing: policy = passing[0]
            status = driver.status(policy)
            budget.mandatory_complete = status in ('ACCEPTED', 'ACCEPTED WITH EXPLICIT LIMITATIONS')
            status = driver.write_results(policy)
            manifest.update(build_manifest(reserves))
            manifest.update(status=status if gate or status == 'FAILED' else 'INCONCLUSIVE', phase_status=status,
                            selected_policy=policy, expected_phase_keys=driver.expected_keys, validation_executed=bool(gate),
                            validation_completed=bool(gate) and budget.mandatory_complete,
                            development_finalized=not gate and budget.mandatory_complete)
            store.json('calibration_manifest.json', manifest, update_owned=True)
            store.sync()
        except (BudgetExhausted, ArithmeticError, ValueError) as error:
            if driver:
                if not isinstance(error, BudgetExhausted): driver.phase_failures.append(str(error))
                status = driver.write_results(policy, final_status='INCONCLUSIVE' if isinstance(error, BudgetExhausted) else 'FAILED')
                store.json('calibration_manifest.json', manifest | dict(status=status, error=str(error), expected_phase_keys=driver.expected_keys), update_owned=True)
                return status
            for key in expected: store.row('landscapes.jsonl', dict(landscape=key, status='missing', error=str(error), normalized_gap=None))
            store.json('budget.json', budget.snapshot())
            status = 'INCONCLUSIVE' if isinstance(error, BudgetExhausted) else 'FAILED'
            store.decision(status+'\n'+str(error))
            for spec in specs: store.row('instances.jsonl', asdict(spec) | dict(status='unrecorded_or_interrupted', error=str(error)))
            store.json('calibration_manifest.json', manifest | dict(status=status, error=str(error)), update_owned=True)
            return status
        # Finalization errors cannot leave a published validation capability.
        # Publication is outside reporting recovery: never rewrite finalized
        # artifacts if lock publication itself fails.
        if not gate and budget.mandatory_complete and lock_output:
            create_validation_lock(lock_output, policy=policy, scope='bounded_B00_native_common_physical_beta',
                                   reserves=reserves, budget=budget, development_status=status,
                                   development_output=store.directory)
        return status
