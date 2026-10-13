"""Read-only joins of admitted computations, results, requests and evidence."""
from collections import Counter
from pathlib import Path
from struct import pack
import json

from ..calibration.storage import canonical_hash


def read_results(path, admissions):
    records, previous = [], None
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        record = json.loads(line); p = record['payload']
        if p['sequence'] != i or p['previous'] != previous or canonical_hash(p) != record['sha256']:
            raise ValueError('Numerical result/request journal chain mismatch')
        records.append(record); previous = record['sha256']
    computations = {canonical_hash(e):e for e in admissions}
    requests, resolutions, results, controls = {}, {}, {}, {}
    for record in records:
        p, digest = record['payload'], record['sha256']
        kind = p['kind']
        if kind == 'request':
            requests[digest] = p
        elif kind == 'resolution':
            request = p['request_id']
            if request not in requests or request in resolutions:
                raise ValueError('Missing or repeated request resolution')
            resolutions[request] = p
        elif kind == 'result':
            computation = p['computation_id']
            if computation not in computations or computation in results:
                raise ValueError('Missing admission or repeated computation result')
            if p['status'] not in ('success','failed') or (p['status']=='failed') != bool(p['error']):
                raise ValueError('Invalid computation outcome')
            e=computations[computation]
            if (type(p.get('gradient_started')) is not bool or p['gradient_started'] and not e['derivative']
                or p.get('matrix_exponentials') not in (0,2) or p['matrix_exponentials'] and not e['oracle']):
                raise ValueError('Computation category counters contradict its admission')
            results[computation] = p
        elif kind == 'control':
            controls[digest] = p['control']
        else:
            raise ValueError('Unknown numerical provenance record')
    for rid, resolution in resolutions.items():
        request = requests[rid]; cid = resolution['computation_id']
        if cid is None:
            if not resolution['error'] or resolution['cached']:
                raise ValueError('Unadmitted request lacks interruption evidence')
            continue
        e = computations.get(cid); result = results.get(cid)
        category = 'oracle' if e and e['oracle'] else 'derivative' if e and e['derivative'] else 'native'
        if (not e or not result or category != request['category']
            or any(e[k] != request[k] for k in ('identity','coordinates','landscape','domain'))
            or result['sequence'] >= resolution['sequence']):
            raise ValueError('Request/result objective, coordinates, category or ordering mismatch')
        if resolution['cached']:
            if request['fresh'] or category != 'native' or result['status'] != 'success' or result['sequence'] >= request['sequence']:
                raise ValueError('Invalid cache provenance')
        elif e.get('request_id') != rid or e.get('fresh') != request['fresh']:
            raise ValueError('A request reused another admission')
        if (result['status']=='failed') != bool(resolution['error']):
            raise ValueError('Request status differs from computation result')
    for cid, e in computations.items():
        if e['oracle'] and e['derivative']:
            raise ValueError('Conflicting computation categories')
        if e.get('request_id') and e['request_id'] not in requests:
            raise ValueError('Admission lacks its objective request')
    return dict(records=records, computations=computations, requests=requests,
                resolutions=resolutions, results=results, controls=controls)


def reconcile(rows, provenance):
    """No numerical backend calls; a digest chain alone cannot pass this join."""
    computations, results = provenance['computations'], provenance['results']
    requests, resolutions = provenance['requests'], provenance['resolutions']
    used_controls, used_work_requests, used_work_computations = set(), set(), set()

    def calculation(cid, identity, arm, point, category, *, value=None, fresh=False):
        e = computations.get(cid); r = results.get(cid)
        actual_category = 'oracle' if e and e['oracle'] else 'derivative' if e and e['derivative'] else 'native'
        if (not e or not r or r['status'] != 'success' or actual_category != category
            or e['identity'] != identity or e['landscape'] != arm or e['domain'] != 'B00'
            or e['coordinates'] != pack('>dd', *map(float,point)).hex()
            or fresh and not e.get('fresh')):
            raise ValueError('Evidence lacks matching successful numerical admission/result')
        if value is not None and r['value'] != value:
            raise ValueError('Evidence contradicts its computation result')
        return r['value']

    def native_request(rid, cid, identity, arm, point, value, *, fresh=False):
        q = requests.get(rid); resolution = resolutions.get(rid)
        if (not q or not resolution or resolution['error'] or resolution['computation_id'] != cid
            or q['fresh'] != fresh or q['category'] != 'native'):
            raise ValueError('Candidate has missing/substituted objective request')
        result = calculation(cid, identity, arm, point, 'native', fresh=fresh)
        if result['gap'] != value:
            raise ValueError('Candidate value differs from its admitted computation')

    def work_links(work, identity, arm, *, allowed_identities=None):
        allowed = {identity} if allowed_identities is None else allowed_identities
        qids, cids = work.get('requests',[]), work.get('computations',[])
        if (len(qids)!=len(set(qids)) or len(cids)!=len(set(cids))
            or used_work_requests.intersection(qids) or used_work_computations.intersection(cids)):
            raise ValueError('Repeated work substituted for a missing attempt/control')
        used_work_requests.update(qids); used_work_computations.update(cids)
        expected = []
        for rid in qids:
            q=requests.get(rid); resolution=resolutions.get(rid)
            if not q or q['identity'] not in allowed or q['landscape']!=arm or q['domain']!='B00':
                raise ValueError('Work request belongs to another objective/arm')
            if resolution and resolution['computation_id'] and not resolution['cached']:
                expected.append(resolution['computation_id'])
            elif not resolution:
                # An interrupted backend may leave an admission without a result.
                expected.extend(cid for cid,e in computations.items() if e.get('request_id')==rid)
        for cid in cids:
            e=computations.get(cid)
            if not e or e['identity'] not in allowed or e['landscape']!=arm or e['domain']!='B00':
                raise ValueError('Work computation belongs to another objective/arm')
            if e['derivative']:
                expected.append(cid)
        if Counter(cids)!=Counter(expected):
            raise ValueError('Recorded attempt/control work omits or invents admissions')
        return [requests[rid] for rid in qids]

    for row in rows:
        for arm_name, arm_row in row['arms'].items():
            bound=arm_row.get('bindings')
            if not bound:
                continue
            identity=bound['objective_identity'];arm=f"{row['family']}:{row['replicate']}:{arm_name}"
            reference=arm_row.get('reference') or {}
            for c in reference.get('candidates',[]):
                native_request(c.get('request_id'),c.get('computation_id'),identity,arm,(c['u'],c['beta']),
                               c['value'],fresh=c.get('observation')=='terminal')
            for attempt in reference.get('attempts',[]):
                work=attempt.get('work')
                if work is None:
                    raise ValueError('Attempt lacks its charged-work provenance')
                work_links(work,identity,arm)
                if attempt.get('actual_evaluations')!=len(work['computations']):
                    raise ValueError('Attempt evaluation count differs from admissions')
                for c in reference.get('candidates',[]):
                    if c.get('attempt_id')==attempt.get('attempt_id') and c['request_id'] not in work['requests']:
                        raise ValueError('Attempt observation was evaluated outside that attempt')
                if attempt.get('failed_request_id'):
                    resolution=resolutions.get(attempt['failed_request_id'])
                    if (attempt['failed_request_id'] not in work['requests'] or not resolution or not resolution['error']
                        or resolution['error']['type']!=attempt.get('interrupted')):
                        raise ValueError('Attempt interruption differs from its failed request')
                if attempt.get('unreturned_request_id'):
                    rid=attempt['unreturned_request_id'];resolution=resolutions.get(rid)
                    if rid not in work['requests'] or not resolution or resolution['error']:
                        raise ValueError('Post-computation interruption lacks its completed request')
            for control in arm_row.get('controls',[]):
                cid=control.get('control_id')
                core={k:v for k,v in control.items() if k not in ('control_id','bindings')}
                if cid in used_controls or provenance['controls'].get(cid)!=core:
                    raise ValueError('Control result missing, altered or reused')
                used_controls.add(cid)
                point=control['point'];kind=control['kind'];work=control['work']
                allowed={identity,control['transported_identity']} if kind=='cyclic_transport' else {identity}
                qs=work_links(work,identity,arm,allowed_identities=allowed)
                # Failed controls remain retained, and cannot qualify any reference.
                if control['status']!='pass':
                    continue
                coords=pack('>dd',*map(float,point)).hex()
                if kind=='candidate_reevaluation':
                    if len(qs)!=1 or not qs[0]['fresh'] or qs[0]['coordinates']!=coords:
                        raise ValueError('Fresh check omitted its fresh objective request')
                    value=calculation(control['checked_computation_id'],identity,arm,point,'native',fresh=True)
                    if len(work['computations'])!=1 or value['gap']!=control['checked_value']:
                        raise ValueError('Fresh check result does not match its admitted computation')
                elif kind=='direct_expm':
                    if (len(qs)!=2 or Counter(q['category'] for q in qs)!=Counter(native=1,oracle=1)
                        or any(q['coordinates']!=coords for q in qs)):
                        raise ValueError('Direct oracle omitted native/oracle work')
                    a=calculation(control['native_computation_id'],identity,arm,point,'native')
                    b=calculation(control['oracle_computation_id'],identity,arm,point,'oracle',fresh=True)
                    if control['errors']['gap']!=abs(a['gap']-b['gap']):
                        raise ValueError('Direct oracle gap discrepancy differs from computed results')
                    if control.get('native',{}).get('normalized_gap')!=a['gap']:
                        raise ValueError('Direct oracle native summary differs from computation result')
                elif kind=='gradient':
                    ds=[cid for cid in work['computations'] if computations[cid]['derivative']]
                    if len(ds)!=1:
                        raise ValueError('Gradient control omitted its derivative evolution')
                    calculation(ds[0],identity,arm,point,'derivative',value=control['analytic'])
                    expected=[coords,coords];values=[]
                    for axis,step in enumerate(control['steps']):
                        sides=[]
                        for sign in (1,-1):
                            p=list(point);p[axis]+=sign*step
                            xy=pack('>dd',*map(float,p)).hex();expected.append(xy)
                            matching=[rid for rid in work['requests'] if requests[rid]['coordinates']==xy]
                            if len(matching)!=1 or not resolutions.get(matching[0]):
                                raise ValueError('Finite difference omitted a side evaluation')
                            comp=resolutions[matching[0]]['computation_id']
                            sides.append(calculation(comp,identity,arm,p,'native')['gap'])
                        values.append((sides[0]-sides[1])/(2*step))
                    if (Counter(q['coordinates'] for q in qs)!=Counter(expected)
                        or any(q['category']!='native' or q['fresh'] for q in qs) or values!=control['numerical']):
                        raise ValueError('Gradient finite-difference work/results mismatch')
                elif kind=='cyclic_transport':
                    if (len(qs)!=2 or Counter(q['identity'] for q in qs)!=Counter([identity,control['transported_identity']])
                        or any(q['category']!='native' or q['coordinates']!=coords for q in qs)):
                        raise ValueError('Transport control omitted a transported/native evaluation')
                    a=calculation(control['native_computation_id'],identity,arm,point,'native')
                    b=calculation(control['transported_computation_id'],control['transported_identity'],arm,point,'native')
                    if control['error']<abs(a['gap']-b['gap']):
                        raise ValueError('Transport control contradicts its computed gaps')
                else:
                    raise ValueError('Unknown numerical control category')
