"""Recalculate v2 decisions from scalar evidence; no QAOA evaluations here."""
from itertools import product
from math import isfinite

import numpy as np

from ..calibration.config import AUDIT_GRAPHS, BY_NAME, de_seed, grid_axes
from ..calibration.decisions import (Check, DEFICIT, DISTORTION, agreement, band,
    contrast, deficit, distortion, reference_status)


def classify(checks, expected):
    observed = {c.name: c for c in checks}
    if len(observed) != len(checks):
        raise ValueError('Duplicate v2 acceptance check')
    if any(c.essential and c.status == 'fail' for c in checks):
        return 'FAILED'
    if not expected or any(n not in observed or observed[n].status != 'pass' for n in expected):
        return 'INCONCLUSIVE'
    return 'QUALIFIED'


def reference_values(ref):
    """Independent routes are derived from candidate provenance, never complete flags."""
    if not ref:
        return None, False
    candidates = ref.get('candidates', [])
    valid = [c for c in candidates if BY_NAME['B00'].contains((c['u'], c['beta']))
             and c.get('value') is not None and isfinite(c['value'])]
    if len(valid) != len(candidates) or not valid:
        return None, False
    a = min((c['value'] for c in valid if c['route'] == 'A'), default=None)
    b = min((c['value'] for c in valid if c['route'] == 'B'), default=None)
    pooled = min(c['value'] for c in valid)
    r = agreement(a, b, pooled)
    f = deficit(ref.get('coarse'), pooled) if ref.get('refined') else None
    consistent = (ref.get('route_a') == a and ref.get('route_b') == b and
        ref.get('pooled') == pooled and ref.get('r') == r and ref.get('f') == f)
    resolved = (consistent and ref.get('faces_complete') is True and
                reference_status(r) == 'pass' and
                (not ref.get('refined') or reference_status(f) == 'pass'))
    return pooled, resolved


def near_nonflat(ref):
    if not ref or not ref.get('candidates'):
        return False
    best = min(ref['candidates'], key=lambda c: (c['value'], c['u'], c['beta']))
    for axis, distances in enumerate(zip(*BY_NAME['B00'].face_distances((best['u'], best['beta'])))):
        for fixed, distance in zip(BY_NAME['B00'].bounds[axis], distances):
            if distance <= .01 and f'{axis}:{fixed}' not in ref.get('flat_faces', []):
                return True
    return False


def _control_ok(c):
    kind = c.get('kind')
    if c.get('status') != 'pass':
        return False
    if kind == 'gradient':
        if len(c.get('analytic',[]))!=2 or len(c.get('numerical',[]))!=2:return False
        error = max(abs(a-b) for a,b in zip(c['analytic'],c['numerical']))
        return error == c['error'] and error <= 1e-6
    if kind=='direct_expm' and set(c.get('errors',{}))!={'probability','gap','normalization','native_normalization','native_state_norm'}:
        return False
    fields = {'direct_expm': tuple(c.get('errors', {}).values()),
              'mapped_spectral_coordinates': (c.get('probability_error'), c.get('gap_error')),
              'candidate_reevaluation': (c.get('error'),),
              'cost_units': (c.get('error'),), 'cyclic_transport': (c.get('error'),)}
    if kind == 'cyclic_reconstruction':
        return isfinite(c.get('reconstructed_gap_sensitivity', float('nan')))
    values = fields.get(kind, ())
    return bool(values) and all(v is not None and isfinite(v) and 0 <= v <= 1e-10 for v in values)


def controls_complete(row):
    key = row['canonical']; i,g,init = key
    width = row['width']
    margins = np.array((1e-5, 1e-5/max(1., width)))
    points = np.random.default_rng(820000000+10000*i+100*g+10*init).uniform(
        margins, np.array((4*np.pi,2*np.pi))-margins, size=(8,2))
    controls = row.get('controls', [])
    if any(c.get('kind')=='gradient' and tuple(c.get('steps',()))!=(1e-5,1e-5/max(1.,width)) for c in controls):
        return False
    required = [('direct_expm', tuple(p)) for p in points]
    required += [('gradient',tuple(p)) for p in points]
    required += [('direct_expm',p) for p in product((0.,2*np.pi,4*np.pi),(0.,np.pi,2*np.pi))]
    grid = tuple(product(np.linspace(0,2*np.pi,5),np.linspace(0,np.pi,5)))
    required += [('mapped_spectral_coordinates',p) for p in grid]
    prod = row.get('production', {}).get('33')
    ref = row.get('references', {}).get('B00')
    if not prod or not ref or not ref.get('candidates'):
        return False
    selected = min(ref['candidates'], key=lambda c:(c['value'],c['u'],c['beta']))
    for candidate in (prod,selected):
        if not any(c.get('kind')=='direct_expm' and tuple(c.get('point',()))==(candidate['u'],candidate['beta'])
            and _control_ok(c) and c.get('native',{}).get('normalized_gap') is not None
            and abs(c['native']['normalized_gap']-candidate['value'])<=1e-10 for c in controls):
            return False
    required += [('direct_expm',(prod['u'],prod['beta'])),
                 ('direct_expm',(selected['u'],selected['beta'])),
                 ('candidate_reevaluation',(selected['u'],selected['beta']))]
    if row['landscape'][0] in (0,2,4,6) or row.get('synthetic'):
        required += [(kind,p) for kind in ('cost_units','cyclic_transport','cyclic_reconstruction') for p in grid]
        for multiplier in (2.**-20,2.**20):
            for point in grid:
                if not any(c.get('kind') == 'cost_units' and c.get('multiplier') == multiplier
                    and tuple(c.get('point',())) == point and _control_ok(c) for c in controls):
                    return False
    return all(any(c.get('kind') == kind and tuple(c.get('point',())) == tuple(point)
                   and _control_ok(c) for c in controls) for kind,point in required)


def required_work(row, paired_trigger=False):
    ref = row.get('references',{}).get('B00')
    if not ref:
        return True
    prod = row.get('production',{}).get('33')
    numerical = (reference_status(ref.get('r')) != 'pass' or not ref.get('faces_complete')
        or prod is None or band(deficit(prod['value'],ref.get('pooled')),DEFICIT) != 'pass'
        or near_nonflat(ref) or paired_trigger)
    audit = row.get('audit') is True
    # F disagreement cannot be resolved by a second mesh; it stays unresolved.
    return ((numerical or audit) and not ref.get('refined') or
            numerical and not ref.get('full_bundle'))


def assess(rows, keys, *, stopped=False, phase_failures=()):
    bykey = {tuple(r['landscape']):r for r in rows}
    if len(bykey) != len(rows) or not set(bykey) <= set(keys):
        raise ValueError('Duplicate or unexpected landscape evidence')
    checks = []; pairs = []; triggers = set()
    values = {}
    for key in keys:
        row = bykey.get(key)
        if row is not None:
            ref = row.get('references',{}).get('B00')
            values[key] = reference_values(ref)
    strata = sorted({(i,init) for i,_,init in keys})
    from itertools import combinations
    for i,init in strata:
        for a,b in combinations(sorted(g for j,g,s in keys if (j,s)==(i,init)),2):
            left,right = bykey.get((i,a,init),{}),bykey.get((i,b,init),{})
            pa=left.get('production',{}).get('33',{}).get('value')
            pb=right.get('production',{}).get('33',{}).get('value')
            ra,ca=values.get((i,a,init),(None,False));rb,cb=values.get((i,b,init),(None,False))
            k=distortion(pa,pb,ra if ca else None,rb if cb else None)
            status=band(k,DISTORTION)
            pairs.append(dict(instance_id=i,initialization_id=init,i=a,j=b,policy=33,
                production_contrast=contrast(pa,pb),reference_contrast=contrast(ra,rb) if ca and cb else None,K=k,status=status))
            checks.append(Check(f'pair:{i}:{init}:{a}:{b}',status))
            if status != 'pass':
                triggers.update((tuple(left.get('canonical',(i,a,init))),tuple(right.get('canonical',(i,b,init)))))
    for key in keys:
        prefix=':'.join(map(str,key));row=bykey.get(key)
        if row is None or row.get('status') != 'observed':
            checks.append(Check(prefix+':coverage','missing'));continue
        prod=row.get('production',{}).get('33'); ref=row.get('references',{}).get('B00')
        r,complete=values.get(key,(None,False))
        controls=row.get('controls',[])
        arithmetic=row.get('arithmetic_failure') or any(c.get('status')=='fail' for c in controls)
        checks += [Check(prefix+':arithmetic','fail' if arithmetic else 'pass'),
                   Check(prefix+':policy','pass' if set(row.get('production',{}))=={'33'} else 'unresolved'),
                   Check(prefix+':controls','pass' if controls_complete(row) else 'unresolved'),
                   Check(prefix+':reference','pass' if complete else 'unresolved'),
                   Check(prefix+':E',band(deficit(prod['value'] if prod else None,r if complete else None),DEFICIT)),
                   Check(prefix+':trigger_work','unresolved' if required_work(row,tuple(row['canonical']) in triggers) else 'pass'),
                   Check(prefix+':execution','unresolved' if row.get('error') else 'pass')]
        for j,proof in enumerate(row.get('negative_evidence',[])):
            values=proof['values']
            for v in values:
                if (not v['faces_complete'] or reference_status(agreement(v['a'],v['b'],v['r']))!='pass'
                    or v['refined'] and reference_status(deficit(v['coarse'],v['r']))!='pass'):
                    raise ValueError('Unresolved evidence cannot establish a completed failure')
            if proof['kind']=='E' and len(values)==1:
                q=deficit(values[0]['production'],values[0]['r']);threshold=DEFICIT
            elif proof['kind']=='K' and len(values)==2:
                a,b=values;q=distortion(a['production'],b['production'],a['r'],b['r']);threshold=DISTORTION
            else:raise ValueError('Invalid preserved counterexample')
            if band(q,threshold)!='fail':raise ValueError('Preserved counterexample is not a clear failure')
            checks.append(Check(prefix+f':completed_failure:{j}','fail'))
    checks.append(Check('resource_completion','unresolved' if stopped else 'pass'))
    checks += [Check(f'phase_failure:{j}', 'fail') for j,_ in enumerate(phase_failures)]
    return classify(checks,[c.name for c in checks]),checks,pairs
