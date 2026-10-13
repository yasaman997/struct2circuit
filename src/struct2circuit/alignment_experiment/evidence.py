"""Scalar/provenance verification and the sole frozen 32-unit decision."""
from collections import Counter
from itertools import product
from math import fsum, isfinite, log, sqrt
from statistics import median

import numpy as np

from ..calibration.config import BY_NAME, grid_axes, numerical_settings
from ..calibration.objective import array_hash
from ..calibration.storage import canonical_hash
from ..mixers import MixerSpec, xy_mixer_hamiltonian
from .config import DESIGN_HASH, seed
from .reference import interior, candidate_id


def bindings(problem, edges, landscape, implementation):
    basis = problem.feasible_basis()
    costs = np.array([fsum([float(problem.Q[i, j]) for i in np.flatnonzero(x) for j in np.flatnonzero(x)]
                          + [float(problem.c[i]) for i in np.flatnonzero(x)]) for x in basis])
    span = float(np.max(costs)-np.min(costs))
    normalized = (costs-np.min(costs))/span if span > 0 else np.array([])
    graph = MixerSpec('c6', 6, tuple(map(tuple, edges)), 'relabeling')
    h = xy_mixer_hamiltonian(basis, graph)
    state = np.full(20, 1/np.sqrt(20), dtype=complex)
    objective = array_hash(problem.Q, problem.c, basis, h, state, normalized)+':'+str(landscape)+':compensated-normalized-p1-v1'
    return dict(objective_identity=objective, coefficients_sha256=array_hash(problem.Q, problem.c),
                graph_sha256=canonical_hash(edges), state_sha256=array_hash(state), design_sha256=DESIGN_HASH,
                implementation=implementation, domain='B00', initialization='uniform', depth=1)


def classify(delta):
    if delta is None:
        return 'unknown'
    if not isfinite(delta):
        raise ValueError('Nonfinite paired outcome')
    return 'win' if delta < -.00012 else 'loss' if delta > .00012 else 'neutral'


def primary(counts, *, integrity_failure=False):
    if set(counts) != {'win', 'loss', 'neutral', 'unknown'} or any(type(v) is not int or v < 0 for v in counts.values()) or sum(counts.values()) != 32:
        raise ValueError('Exactly 32 independent QUBO contrasts must remain')
    w, u = counts['win'], counts['unknown']
    eps = sqrt(log(40)/64)
    return dict(counts=counts, denominator=32, lower=max(0, w/32-eps), upper=min(1, (w+u)/32+eps),
                outcome='INCONCLUSIVE' if integrity_failure else 'DIRECTIONAL_SUPPORT' if w >= 24 else
                        'CONTRADICTION' if w+u <= 8 else 'INCONCLUSIVE', integrity_failure=integrity_failure)


def _starts(candidates, axes, count):
    chosen = []
    for c in sorted(candidates, key=lambda c: (c['value'], c['u'], c['beta'])):
        point = (c['u'], c['beta'])
        index = tuple(int(np.searchsorted(axes[i], point[i])) for i in (0, 1))
        if all(max(abs(index[i]-j[i]) for i in (0, 1)) >= 2 for j in [d[1] for d in chosen]):
            chosen.append((point, index))
        if len(chosen) == count:
            break
    return [list(p) for p, _ in chosen]


def verify_attempts(reference, objective_identity):
    """Bind solver observations and terminal fresh checks to the exact attempt."""
    candidates = reference.get('candidates', [])
    by_id = {c.get('candidate_id'): c for c in candidates}
    if None in by_id or len(by_id) != len(candidates):
        return False
    for c in candidates:
        if c['candidate_id'] != candidate_id(c) or not c.get('request_id') or not c.get('computation_id'):
            raise ValueError('Candidate provenance digest or numerical link mismatch')
    complete = True
    attempts = reference.get('attempts', [])
    ids = set()
    solver_sources = {'lbfgsb','fine_lbfgsb','de_0','de_1','de_polish_0','de_polish_1','powell'}
    for index, a in enumerate(attempts):
        fields = ('objective_identity','index','route','source','seed','start','start_candidate_id',
                  'bounds','settings','objective_call_cap')
        if any(k not in a for k in fields) or not a.get('attempt_id'):
            complete = False
            continue
        if (a['index'] != index or a['objective_identity'] != objective_identity
            or a['attempt_id'] != canonical_hash({k:a[k] for k in fields}) or a['attempt_id'] in ids):
            raise ValueError('Attempt specification/identity mismatch')
        ids.add(a['attempt_id'])
        observed = [c for c in candidates if c.get('attempt_id') == a['attempt_id']]
        solver = [c for c in observed if c.get('observation') == 'solver']
        terminal = [c for c in observed if c.get('observation') == 'terminal']
        if (any(c['source'] != a['source'] or c['route'] != a['route'] or c.get('seed') != a['seed'] for c in observed)
            or len(observed) != len(solver)+len(terminal)):
            raise ValueError('Candidate attempt route/source/seed substitution')
        if ([c['candidate_id'] for c in solver] != a.get('observations')
            or a.get('objective_calls') != len(solver)+int(bool(a.get('failed_request_id')))
                                          +int(bool(a.get('unreturned_request_id')))+int(bool(a.get('blocked_call')))):
            complete = False
        if not solver or not a.get('work'):
            complete = False
        if len({c['request_id'] for c in observed}) != len(observed):
            raise ValueError('A solver observation reused another request')
        if a.get('interrupted') is None:
            if len(terminal) != 1 or terminal[0]['candidate_id'] != a.get('terminal_candidate_id') or a.get('failed_request_id'):
                complete = False
            elif type(a.get('success')) is not bool or terminal[0].get('success') != a['success']:
                raise ValueError('Terminal observation differs from the solver return status')
        else:
            complete = False  # retained partial work never establishes completeness
            if terminal or a.get('terminal_candidate_id'):
                raise ValueError('Interrupted attempt claims a terminal evaluation')
            if a['interrupted'] == 'AttemptLimit' and (
                a.get('message') != 'solver_objective_call_cap' or a['objective_calls'] != a['objective_call_cap']
                or a.get('failed_request_id') or a.get('unreturned_request_id') or a.get('blocked_call')):
                raise ValueError('AttemptLimit without exhausting the declared allowance')
            if (a.get('unreturned_request_id') or a.get('blocked_call')) and a['interrupted'] not in ('BudgetExhausted','ArmExhausted'):
                raise ValueError('Unreturned/blocked request lacks a resource interruption')
        if a['source'].startswith('de_polish_'):
            source = 'de_'+a['source'][-1]
            de = [c for c in candidates if c['source'] == source]
            winner = min(de, key=lambda c:(c['value'],c['u'],c['beta'])) if de else None
            if not winner:
                complete = False
            elif a['start_candidate_id'] != winner['candidate_id'] or a['start'] != [winner['u'],winner['beta']]:
                raise ValueError('DE polish start differs from its retained DE winner')
    if any(c['source'] in solver_sources and c.get('attempt_id') not in ids for c in candidates):
        complete = False
    pre_powell = {}
    for c in sorted((c for c in candidates if c['source'] != 'powell'), key=lambda c:(c['value'],c['u'],c['beta'])):
        pre_powell.setdefault((c['u'],c['beta']),c)
    for a, start in zip((a for a in attempts if a['source']=='powell'), list(pre_powell.values())[:2]):
        if a.get('start_candidate_id') != start['candidate_id'] or a['route'] != start['route']:
            raise ValueError('Powell did not inherit its selected starting candidate lineage')
    return complete


def verify_reference(row, family, replicate):
    """No objective calls. Flags alone cannot establish numerical completeness."""
    bound = row['bindings']
    reference = row.get('reference')
    controls = row.get('controls', [])
    integrity = bool(row.get('integrity_failure'))
    for control in controls:
        if control.get('bindings') != bound or control.get('domain') != 'B00':
            raise ValueError('Missing/substituted numerical-control objective identity')
        if integrity or control.get('status') == 'fail':
            integrity = True
            continue  # failed/nonfinite raw controls remain visible; no positive inference
        values = list(control.get('errors', {}).values()) if 'errors' in control else [control.get('error')]
        if not values or any(v is None or not isfinite(v) or v < 0 for v in values):
            raise ValueError('Invalid numerical control error')
        tol = 1e-6 if control['kind'] == 'gradient' else 1e-10
        if control['kind'] == 'direct_expm' and set(control.get('errors', {})) != {
                'probability', 'gap', 'normalization', 'native_normalization', 'native_state_norm'}:
            raise ValueError('Direct oracle omitted a required error component')
        if control['kind'] == 'gradient':
            analytic, numerical = np.asarray(control.get('analytic', [])), np.asarray(control.get('numerical', []))
            if (analytic.shape != (2,) or numerical.shape != (2,) or not np.isfinite(analytic).all()
                or not np.isfinite(numerical).all() or float(np.max(abs(analytic-numerical))) != control['error']):
                raise ValueError('Gradient control scalar inconsistency')
            if control.get('steps') != [1e-5, 1e-5/max(1., row['mixer_width'])]:
                raise ValueError('Finite-difference control steps changed')
        if control['kind'] == 'candidate_reevaluation' and (
                control.get('checked_value') is None or abs(control['checked_value']-control['value']) != control['error']):
            raise ValueError('Fresh candidate reevaluation scalar inconsistency')
        if any(v > tol for v in values) or control.get('status') != 'pass':
            integrity = True
    if integrity:
        return None, True
    if not reference:
        return None, integrity
    candidates = reference.get('candidates', [])
    for c in candidates:
        if (c.get('objective_identity') != bound['objective_identity'] or c.get('route') not in ('A', 'B')
            or c.get('source') not in ('grid', 'fine_grid', 'face_grid', 'face_scalar', 'lbfgsb',
                                      'fine_lbfgsb', 'de_0', 'de_1', 'de_polish_0', 'de_polish_1', 'powell')
            or not BY_NAME['B00'].contains((c.get('u'), c.get('beta')))
            or not isfinite(c['value'])):
            raise ValueError('Substituted or invalid reference candidate')
        if c['source'] in ('grid', 'fine_grid', 'face_grid', 'face_scalar', 'lbfgsb', 'fine_lbfgsb') and c['route'] != 'A':
            raise ValueError('Grid/local candidates cannot manufacture the independent DE route')
        if c['source'].startswith('de_') and (c['route'] != 'B' or c.get('seed') != seed('de1' if c['source'].endswith('1') else 'de0', family, replicate)):
            raise ValueError('DE candidate lineage/seed mismatch')
    mandatory = verify_attempts(reference, bound['objective_identity'])
    mandatory = mandatory and not row.get('unresolved_reason') and bool(reference.get('refined')) and bool(reference.get('full_bundle'))
    for level, source, count in [(0, 'grid', 4), (1, 'fine_grid', 8)]:
        axes = grid_axes(BY_NAME['B00'], 128, level)
        grid = [c for c in candidates if c['source'] == source]
        expected = set(product(*axes))
        if len(grid) != len(expected) or {(c['u'], c['beta']) for c in grid} != expected:
            mandatory = False
        attempts = [a for a in reference.get('attempts', []) if a['source'] == ('lbfgsb' if level == 0 else 'fine_lbfgsb')]
        if [a.get('start') for a in attempts] != _starts(grid, axes, count):
            mandatory = False
    attempts = reference.get('attempts', [])
    required = {'lbfgsb': 4, 'fine_lbfgsb': 8, 'de_0': 1, 'de_1': 1,
                'de_polish_0': 1, 'de_polish_1': 1, 'powell': 2}
    if Counter(a['source'] for a in attempts) != Counter(required):
        mandatory = False
    powell = [a for a in attempts if a['source'] == 'powell']
    if len({tuple(a.get('start', [])) for a in powell}) != 2:
        mandatory = False
    pre_powell = {}
    for c in sorted((c for c in candidates if c['source'] != 'powell'), key=lambda c: (c['value'], c['u'], c['beta'])):
        pre_powell.setdefault((c['u'], c['beta']), c)
    if [a.get('start') for a in powell] != [list(p) for p in list(pre_powell)[:2]]:
        mandatory = False
    settings = numerical_settings()
    for a in attempts:
        source = a['source']
        cap = 1024 if source == 'de_0' else 2032 if source == 'de_1' else 250
        kind = 'de' if source in ('de_0', 'de_1') else 'powell' if source == 'powell' else 'local'
        if (canonical_hash(a.get('bounds')) != canonical_hash(BY_NAME['B00'].bounds) or canonical_hash(a.get('settings')) != canonical_hash(settings[kind])
            or a.get('objective_call_cap') != cap or not 0 < a.get('objective_calls', 0) <= cap):
            raise ValueError('Reference solver settings/cap mismatch')
        if source.startswith('de_'):
            role = 'de1' if source.endswith('1') else 'de0'
            if a.get('seed') != seed(role, family, replicate) or a.get('route') != 'B':
                raise ValueError('DE seed or independent route substituted')
        elif source != 'powell' and a.get('route') != 'A':
            raise ValueError('Grid local attempt route changed')
        if a.get('interrupted') is not None:
            mandatory = False
    faces = row.get('faces', [])
    wanted = {(level, axis, endpoint) for level in (0, 1) for axis in (0, 1) for endpoint in BY_NAME['B00'].bounds[axis]}
    if len(faces) != 8 or {(f['level'], f['axis'], f['fixed']) for f in faces} != wanted:
        mandatory = False
    for face in faces:
        axes = grid_axes(BY_NAME['B00'], 128, face['level'])
        expected = [(face['fixed'], float(x)) if face['axis'] == 0 else (float(x), face['fixed']) for x in axes[1-face['axis']]]
        if face['points'] != [list(p) for p in expected] or face['complete'] is not True:
            mandatory = False
        values = face.get('values', [])
        if len(values) != len(expected) or any(not isfinite(v) for v in values):
            mandatory = False
            continue
        grid_source = 'grid' if face['level'] == 0 else 'fine_grid'
        grid_values = {(c['u'], c['beta']): c['value'] for c in candidates if c['source'] == grid_source}
        if any(p not in grid_values or grid_values[p] != v for p, v in zip(expected, values)):
            mandatory = False
        flat = face['axis'] == 1 and face['fixed'] == 0
        # Uniform C6 has unequal feasible degrees (2,4,6), so it is not an
        # eigenstate; no other face can use the eigenstate flatness shortcut.
        if face['flat'] != flat:
            mandatory = False
        brackets = [[expected[i-1][1-face['axis']], expected[i+1][1-face['axis']]]
                    for i in range(1, len(values)-1) if values[i] < min(values[i-1], values[i+1])]
        actual = face.get('brackets', [])
        if ([b['bounds'] for b in actual] != ([] if flat else brackets)
            or any(not b['complete'] or not 0 < b['calls'] <= 60 for b in actual)):
            mandatory = False
        by_id = {c.get('candidate_id'):c for c in candidates}
        for bracket in actual:
            observations = bracket.get('observations', [])
            if len(observations)!=bracket['calls'] or len(set(observations))!=len(observations):
                mandatory = False
            for cid in observations:
                c=by_id.get(cid)
                if (not c or c['source']!='face_scalar' or c['route']!='A'
                    or (c['u'],c['beta'])[face['axis']]!=face['fixed']
                    or not bracket['bounds'][0] <= (c['u'],c['beta'])[1-face['axis']] <= bracket['bounds'][1]):
                    raise ValueError('Boundary bracket observation substituted or missing')
    if not mandatory:
        return None, integrity
    if not all(any(c['route'] == route for c in candidates) for route in ('A', 'B')):
        return None, integrity
    a, b = [min(c['value'] for c in candidates if c['route'] == route) for route in ('A', 'B')]
    r = min(c['value'] for c in candidates)
    R = max(abs(a-b), abs(a-r), abs(b-r))
    coarse = reference.get('coarse')
    coarse_evidence = (row.get('coarse_reference') or {}).get('candidates', [])
    final_counts = Counter(canonical_hash(c) for c in candidates)
    coarse_counts = Counter(canonical_hash(c) for c in coarse_evidence)
    if (not coarse_evidence or any(final_counts[k] < n for k,n in coarse_counts.items())
        or any(c['source'] in ('fine_grid','fine_lbfgsb','de_1','de_polish_1','powell') for c in coarse_evidence)
        or coarse != min(c['value'] for c in coarse_evidence)):
        raise ValueError('Pre-refinement value differs from preserved coarse evidence')
    F = coarse-r if coarse is not None else None
    if reference.get('pooled') != r or reference.get('route_a') != a or reference.get('route_b') != b or reference.get('r') != R or reference.get('f') != F:
        raise ValueError('Reference scalar summary differs from stored candidates')
    wanted_controls = [('gradient', tuple(p)) for p in row['interior_points']]
    wanted_controls += [('direct_expm', tuple(p)) for p in row['interior_points']]
    wanted_controls += [('direct_expm', p) for p in product((0., 2*np.pi), (0., np.pi))]
    winners = [min((c for c in candidates if c['route'] == route), key=lambda c: (c['value'], c['u'], c['beta'])) for route in ('A', 'B')]
    winners += [min(candidates, key=lambda c: (c['value'], c['u'], c['beta']))]
    # Bind the three required roles separately, including repeated winners.
    required_fresh = [(kind, role, c['candidate_id']) for role,c in zip(('route_a','route_b','selected'), winners)
                      for kind in ('direct_expm','candidate_reevaluation')]
    actual_fresh = [(c['kind'], c.get('winner_role'), c.get('candidate_id')) for c in controls if c.get('winner_role')]
    if Counter(actual_fresh) != Counter(required_fresh):
        mandatory = False
    by_id = {c['candidate_id']:c for c in candidates}
    for check in controls:
        if check['kind'] == 'candidate_reevaluation' or check.get('winner_role'):
            c = by_id.get(check.get('candidate_id'))
            if c is None or tuple(check['point']) != (c['u'],c['beta']):
                raise ValueError('Fresh control substituted its candidate identity/coordinates')
            if check['kind'] == 'candidate_reevaluation' and (
                check.get('value') != c['value'] or check['error'] != abs(check['checked_value']-c['value'])):
                raise ValueError('Fresh reevaluation differs from the original candidate value')
    wanted_controls += [(kind, (c['u'], c['beta'])) for c in winners for kind in ('direct_expm', 'candidate_reevaluation')]
    wanted_controls += [('cyclic_transport', p) for p in product(np.linspace(0., 2*np.pi, 5), np.linspace(0., np.pi, 5))]
    observed = Counter((c['kind'], tuple(c['point'])) for c in controls)
    if any(observed[k] < n for k, n in Counter(wanted_controls).items()):
        mandatory = False
    return (r if mandatory and R <= 1e-5 and F is not None and 0 <= F <= 1e-5 and not integrity else None), integrity


def analyze(rows):
    if {(r['family'], r['replicate']) for r in rows} != set(product(range(4), range(8))) or len(rows) != 32:
        raise ValueError('Frozen 32-QUBO panel missing, duplicated or substituted')
    counts = dict(win=0, loss=0, neutral=0, unknown=0)
    families = {f: dict(win=0, loss=0, neutral=0, unknown=0) for f in range(4)}
    comparisons, integrity, missingness = [], False, []
    for row in rows:
        values = {}
        for arm in ('low', 'high'):
            value, failed = verify_reference(row['arms'][arm], row['family'], row['replicate'])
            values[arm] = value
            integrity = integrity or failed
            ref = row['arms'][arm].get('reference') or {}
            missingness.append(dict(family=row['family'], replicate=row['replicate'], arm=arm,
                complete=value is not None, R=ref.get('r'), F=ref.get('f'),
                reason=row['arms'][arm].get('unresolved_reason') or (None if value is not None else 'reference_or_required_work_unresolved')))
        delta = values['high']-values['low'] if all(v is not None for v in values.values()) else None
        outcome = classify(delta)
        counts[outcome] += 1; families[row['family']][outcome] += 1
        selection = (row.get('selection') or {}).get('selected', {})
        scores = {a: selection.get(a, {}).get('score') for a in ('low', 'high')}
        route_deltas = {route: row['arms']['high']['reference'][route]-row['arms']['low']['reference'][route]
                        if delta is not None else None for route in ('route_a', 'route_b')}
        comparisons.append(dict(family=row['family'], replicate=row['replicate'], delta=delta,
            status=outcome, references=values, alignment_scores=scores, route_deltas=route_deltas,
            score_separation=scores['high']-scores['low'] if all(v is not None for v in scores.values()) else None))
    for f in range(4):
        eligible = [c['delta'] for c in comparisons if c['family'] == f and c['delta'] is not None]
        families[f].update(eligible_pairs=len(eligible), median_delta=median(eligible) if eligible else None,
                           min_delta=min(eligible) if eligible else None, max_delta=max(eligible) if eligible else None)
    eligible = [c for c in comparisons if c['delta'] is not None and c['score_separation'] is not None]
    x, y = [np.array([c[k] for c in eligible]) for k in ('score_separation', 'delta')]
    correlation = float(np.corrcoef(x,y)[0,1]) if len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0 else None
    return primary(counts, integrity_failure=integrity) | dict(families=families, comparisons=comparisons,
                missingness=missingness, descriptive_association=dict(eligible_pairs=len(eligible),
                pearson=correlation, p_value=None), independent_verification_required=True, qualification='empirical_B00_only')
