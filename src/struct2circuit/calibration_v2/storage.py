"""Independent v2 evidence verification and durable, version-bound capabilities."""
from dataclasses import asdict
from hashlib import sha256
from itertools import product
from pathlib import Path
from struct import pack
import csv
import json
import os
import tempfile
from time import process_time

import numpy as np
from threadpoolctl import threadpool_info

from ..calibration import config as v1
from ..calibration.decisions import THRESHOLDS, band, DEFICIT, DISTORTION
from ..calibration.objective import array_hash
from ..calibration.storage import (ARTIFACTS, canonical_hash, canonical_json, write_canonical,
    sha256_file, preflight_validation_lock, _sync_directory)
from .budget import Budget, Limits, verify_continuation
from .config import HISTORY, PROTOCOL_HASH, RULES, SCOPE, expected_keys, expected_pairs, identity, manifest, specs
from .evidence import assess, required_work, reference_values
from .reference import ReferenceSearch
from ..calibration.reference import Candidate


def atomic_json(path,value):
    path=preflight_validation_lock(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile('w',dir=path.parent,encoding='utf-8',delete=False) as f:
            temporary=Path(f.name);write_canonical(f,value);f.flush();os.fsync(f.fileno())
        os.link(temporary,path);_sync_directory(path.parent)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return path


def file_hash(path):
    path=Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError('Evidence may not use symlink aliases')
    with path.open('rb') as f:return sha256_file(f)


def runtime_identity():
    variables=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
    return dict(environment={n:os.environ.get(n) for n in variables},pools=threadpool_info(),
                software=identity())


def require_official_runtime():
    variables=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
    if any(os.environ.get(n)!='1' for n in variables):
        raise ValueError('Official v2 processes require one numerical thread before imports')
    pools=threadpool_info()
    if not pools or any(p['num_threads']!=1 for p in pools):
        raise ValueError('Loaded numerical pools must use one thread')
    software=identity()
    if tuple(software[n] for n in ('python','numpy','scipy'))!=('3.12.14','2.5.3','1.18.1'):
        raise ValueError('Official v2 execution requires the exact protocol-frozen Python/NumPy/SciPy versions')
    return runtime_identity()


def authorize(path,phase,output,*,lock=None):
    if path is None:raise PermissionError('Separate persisted v2 execution authorization required')
    a=json.loads(Path(path).read_text())
    if (a.get('calibration_execution_authorized') is not True or a.get('protocol_sha256')!=PROTOCOL_HASH
        or a.get('driver')!=identity() or a.get('phase')!=phase or not a.get('approval_identity')
        or a.get('output')!=str(Path(output).absolute())
        or phase=='validation' and (lock is None or a.get('lock_sha256')!=file_hash(lock))):
        raise PermissionError('Authorization does not bind the approved v2 driver, phase and destination')
    if phase=='diagnostics' and (not a.get('core_artifact_hashes') or a.get('diagnostic_design')!=manifest()['diagnostic_domains']):
        raise PermissionError('Diagnostics require the pre-frozen design and qualified core identities')
    archive=a.get('historical_archive')
    if not archive or file_hash(archive)!=HISTORY['archive_sha256']:
        raise PermissionError('Authorization must bind the preserved historical FAILED archive')
    root=Path(__file__).resolve().parents[3]
    if file_hash(root/'paper/numerical_calibration_protocol_v1.md')!=v1.PROTOCOL_HASH:
        raise ValueError('Historical v1 protocol changed')
    return a


def finish_resources(output,budget,phase,*,previous=None):
    """Small immutable cutoff record; the entry supervisor seals process-exit CPU."""
    path=Path(str(output)+'.resources.json').absolute()
    if budget.resource_guard:budget.resource_guard.check(budget,force=True)
    snapshot=budget.snapshot()
    if previous:
        prior=_receipt_snapshot(previous)
        verify_continuation(prior,snapshot)
    elif phase!='development':
        raise ValueError('A later phase must bind its preceding settled resource receipt')
    data=dict(schema_version=2,phase=phase,core_directory=str(Path(output).absolute()),
        core_budget_sha256=file_hash(Path(output)/'budget.json') if (Path(output)/'budget.json').stat().st_size else None,
        counters=snapshot['counters'],phases=snapshot['phases'],per_landscape=snapshot['per_landscape'],
        per_domain=snapshot['per_domain'],cpu_seconds=snapshot['cpu_seconds'],
        wall_seconds=snapshot['wall_seconds'],cpu_cutoff=snapshot['cpu_measurement_cutoff'],
        previous=str(Path(previous).absolute()) if previous else None,previous_sha256=file_hash(previous) if previous else None,
        stop_reason=budget.stop_reason,process_ended=False)
    atomic_json(path,dict(payload=data,sha256=canonical_hash(data)))
    return path


def seal_process(resource_path,*,process_cpu_seconds,supervisor_cpu_seconds=0.,prior_cpu=0.):
    """Called by the entry supervisor after wait4, never a worker's early snapshot."""
    resource_path=Path(resource_path).absolute();data=_record(resource_path)
    if not all(np.isfinite(x) and x>=0 for x in (process_cpu_seconds,supervisor_cpu_seconds,prior_cpu)):
        raise ValueError('Invalid full-process CPU receipt')
    cpu=prior_cpu+process_cpu_seconds+supervisor_cpu_seconds
    if cpu<data['cpu_seconds']:raise ValueError('Process receipt cannot reduce cumulative CPU')
    payload=dict(schema_version=2,previous=str(resource_path),previous_sha256=file_hash(resource_path),
        cpu_seconds=cpu,process_cpu_seconds=process_cpu_seconds,supervisor_cpu_seconds=supervisor_cpu_seconds,
        prior_cpu=prior_cpu,worker_ended=True,process_ended=False,
        stop_reason='cpu_ceiling' if cpu>=Limits().cpu_seconds else data['stop_reason'])
    path=Path(str(resource_path)+'.process-ended.json')
    atomic_json(path,dict(payload=payload,sha256=canonical_hash(payload)))
    return path


def _record(path):
    document=json.loads(Path(path).read_text());p=document['payload']
    if document['sha256']!=canonical_hash(p) or p.get('schema_version')!=2:
        raise ValueError('Invalid v2 resource record')
    if p.get('previous') and file_hash(p['previous'])!=p.get('previous_sha256'):
        raise ValueError('Resource chain was altered')
    return p


def settle_process(resource,*,process_cpu_seconds,publication=None):
    """OS observer records the exited phase coordinator AND its waited children.

    The coordinator has already finished seal_process(), lock publication (if any),
    fsyncs and shutdown. This process-tree CPU replaces early cutoff estimates;
    it is never added to those estimates, which would double-charge the workers.
    """
    resource=Path(resource).absolute();receipt=_record(resource)
    if not np.isfinite(process_cpu_seconds) or process_cpu_seconds<0:
        raise ValueError('Invalid phase-exit CPU measurement')
    cpu=receipt['prior_cpu']+process_cpu_seconds
    if cpu<receipt['cpu_seconds']:
        raise ValueError('Full process-tree CPU cannot reduce a recorded cutoff')
    payload=dict(schema_version=2,previous=str(resource),previous_sha256=file_hash(resource),
        process_ended=True,cpu_seconds=cpu,process_tree_cpu_seconds=process_cpu_seconds,
        cpu_accounting='wait4-exited-coordinator-and-waited-children',
        stop_reason='cpu_ceiling' if cpu>=Limits().cpu_seconds else receipt['stop_reason'])
    if publication is not None:
        pub=Path(str(publication)+'.publication.json');p=_record(pub)
        if p['resource_sha256']!=file_hash(resource) or p['cpu_seconds']>cpu:
            raise ValueError('Process-exit receipt does not cover publication CPU')
        payload.update(publication=str(pub),publication_sha256=file_hash(pub))
    return atomic_json(str(resource)+'.settled.json',dict(payload=payload,sha256=canonical_hash(payload)))


def settle_publication(lock,*,process_cpu_seconds):
    """Settle a separately invoked lock publisher after its complete process exit."""
    pub=Path(str(lock)+'.publication.json');p=_record(pub)
    prior=_receipt_snapshot(p['resource_path'])
    cpu=prior['cpu_seconds']+process_cpu_seconds
    if not np.isfinite(cpu) or process_cpu_seconds<0 or cpu<p['cpu_seconds']:
        raise ValueError('Publisher exit CPU does not cover its finalization cutoff')
    payload=dict(schema_version=2,kind='publication-exit',previous=str(pub.absolute()),previous_sha256=file_hash(pub),
        resource_settlement_sha256=file_hash(str(p['resource_path'])+'.settled.json'),
        cpu_seconds=cpu,process_ended=True,process_tree_cpu_seconds=process_cpu_seconds,
        stop_reason='cpu_ceiling' if cpu>=Limits().cpu_seconds else prior['stop_reason'])
    return atomic_json(str(lock)+'.publication-ended.json',dict(payload=payload,sha256=canonical_hash(payload)))


def _receipt_snapshot(resource,*,_allow_unsettled=False,_seen=None):
    receipt=_record(resource)
    if receipt.get('kind')=='publication-exit':
        pub=_record(receipt['previous'])
        s=_receipt_snapshot(pub['resource_path'],_seen=_seen)
        if (receipt['resource_settlement_sha256']!=file_hash(str(pub['resource_path'])+'.settled.json')
            or receipt['cpu_seconds']<max(s['cpu_seconds'],pub['cpu_seconds'])):
            raise ValueError('Publisher process-exit grant was altered')
        return s | dict(cpu_seconds=receipt['cpu_seconds'],stop_reason=receipt['stop_reason'])
    cutoff=_record(receipt['previous'])
    return current_snapshot(cutoff['core_directory'],resource,_allow_unsettled=_allow_unsettled,_seen=_seen)


def current_snapshot(core,resource,*,_allow_unsettled=False,_seen=None,expected_prior=None):
    core=Path(core);s=json.loads((core/'budget.json').read_text())
    resource=Path(resource).absolute()
    seen=set() if _seen is None else _seen
    key=str(Path(resource).absolute())
    if key in seen:raise ValueError('Cyclic resource chain')
    seen.add(key)
    p=_record(resource)
    if not p.get('worker_ended'):raise ValueError('A full-worker exit receipt is required')
    cutoff=_record(p['previous'])
    if cutoff.get('core_directory')!=str(core.absolute()) or cutoff['core_budget_sha256']!=file_hash(core/'budget.json'):
        raise ValueError('Resource receipt does not bind this core budget')
    if any(canonical_hash(cutoff.get(k))!=canonical_hash(s[k]) for k in
           ('counters','phases','per_landscape','per_domain')) or cutoff['phase']!=s['phase']:
        raise ValueError('Resource and core evaluation ledgers disagree')
    if _allow_unsettled:
        final=p  # Private lock construction only; cannot authorize numerical work.
    else:
        final=_record(str(resource)+'.settled.json')
        if (not final.get('process_ended') or final.get('previous')!=str(resource)
            or final['previous_sha256']!=file_hash(resource) or final['cpu_seconds']<p['cpu_seconds']):
            raise ValueError('Phase sealing/finalization lacks an OS process-exit settlement')
    s.update(cpu_seconds=final['cpu_seconds'],stop_reason=final['stop_reason'])
    if cutoff.get('previous'):
        prior=_receipt_snapshot(cutoff['previous'],_seen=seen)
        if expected_prior is not None and canonical_hash(prior)!=canonical_hash(expected_prior):
            raise ValueError('Preceding resource grant differs from the locked study grant')
        verify_continuation(prior,s)
        if p['prior_cpu']!=prior['cpu_seconds']:
            raise ValueError('CPU receipt does not carry the settled prior-phase grant')
    elif s['phase']!='development' or p['prior_cpu']!=0:
        raise ValueError('A later phase discarded its preceding resource receipt')
    return s


def _rows(path):
    with Path(path).open() as f:
        for line in f:yield json.loads(line)


def verify_evidence(directory,*,phase='development',hashes=None,expected_driver=None):
    """Verify persisted numerical evidence without generating or evaluating a model."""
    require_official_runtime()
    directory=Path(directory)
    actual={n:file_hash(directory/n) for n in ARTIFACTS}
    if hashes is not None and actual!=hashes:raise ValueError('Immutable core evidence was altered')
    m=json.loads((directory/'calibration_manifest.json').read_text())
    if (m.get('schema_version')!=2 or m.get('rules_version')!=RULES or m.get('phase')!=phase
        or m.get('protocol_sha256')!=PROTOCOL_HASH or m.get('policy')!=33 or m.get('domain')!='B00'
        or m.get('history')!=HISTORY or not m.get('reporting_complete')
        or m.get('provenance',{}).get('driver')!=(expected_driver or identity())
        or any(canonical_hash(m.get(k))!=canonical_hash(manifest()[k]) for k in
               ('instances','settings','graph_labels','initializations','development_keys','validation_keys','audits','sentinels'))):
        raise ValueError('Core manifest does not match approved v2')
    if canonical_hash(m.get('runtime'))!=canonical_hash(runtime_identity()):
        raise ValueError('Software, BLAS or thread identity changed')
    s=json.loads((directory/'budget.json').read_text())
    Budget.from_snapshot(s,phase=phase)
    keys_computed={(i,bytes.fromhex(p)) for i,p in s['unique_keys']}
    if s.get('incomplete_required_work')!=[] or not s.get('mandatory_complete'):
        raise ValueError('Mandatory work is incomplete')
    keys=expected_keys(phase)
    instances=list(_rows(directory/'instances.jsonl'))
    if len(instances)!=len(specs(phase)) or {r['instance_id'] for r in instances}!={x.instance_id for x in specs(phase)}:
        raise ValueError('Instance coverage mismatch')
    for row in instances:
        spec=next(x for x in specs(phase) if x.instance_id==row['instance_id'])
        if any(row.get(k)!=v for k,v in asdict(spec).items()) or row.get('status')!='generated':
            raise ValueError('Incorrect instance specification')
        if (row.get('generator')!=v1.FAMILIES[spec.family][0] or row.get('parameters')!=v1.FAMILIES[spec.family][1]
            or np.asarray(row['Q']).shape!=(spec.n,spec.n) or np.asarray(row['c']).shape!=(spec.n,)
            or array_hash(np.asarray(row['Q'],float),np.asarray(row['c'],float))!=row['coefficients_sha256']):
            raise ValueError('Coefficient identity mismatch')
    graphs=list(_rows(directory/'graphs.jsonl'))
    rows=list(_rows(directory/'landscapes.jsonl'))
    for collection in (graphs,rows):
        if len(collection)!=len(keys) or {tuple(r['landscape']) for r in collection}!=set(keys):
            raise ValueError('Landscape coverage mismatch')
    graphmap={tuple(r['landscape']):r for r in graphs}
    rowmap={tuple(r['landscape']):r for r in rows}
    for row in rows:
        key=tuple(row['landscape']);i,g,init=key;graph=graphmap[key]
        canonical=tuple(row['canonical']);spec=next(x for x in specs(phase) if x.instance_id==i)
        if (canonical[0]!=i or canonical[2]!=init or canonical[1]>g or canonical not in rowmap
            or graph['canonical']!=row['canonical'] or row.get('synthetic')
            or row['audit']!=(g==v1.AUDIT_GRAPHS[spec.family]) or row['n_beta']!=graph['n_beta']
            or row['width']!=graph['spectrum']['width'] or set(row['production'])!={'33'}):
            raise ValueError('Alias, audit or production contract mismatch')
        if (graph['graph_label']!=v1.GRAPH_LABELS[g] or len(graph['edges'])!=spec.n
            or array_hash(np.asarray(graph['edges']))!=graph['edge_hash']
            or array_hash(np.asarray(graph['initial_state_real'])+1j*np.asarray(graph['initial_state_imag']))!=graph['initial_state_sha256']):
            raise ValueError('Graph or initial-state identity mismatch')
        alias=rowmap[canonical]
        if canonical!=key and (row['objective_identity']!=alias['objective_identity'] or
            canonical_hash(row['production'])!=canonical_hash(alias['production']) or
            canonical_hash(row['references'])!=canonical_hash(alias['references']) or
            graph['edges']!=graphmap[canonical]['edges'] or graph['initial_state_sha256']!=graphmap[canonical]['initial_state_sha256']):
            raise ValueError('Alias evidence diverges')
        ref=row['references']['B00'];prod=row['production']['33']
        for c in [prod]+ref['candidates']:
            if c['objective_identity']!=row['objective_identity'] or not v1.BY_NAME['B00'].contains((c['u'],c['beta'])):
                raise ValueError('Candidate objective/bounds mismatch')
            if (row['objective_identity'],pack('>dd',c['u'],c['beta'])) not in keys_computed:
                raise ValueError('Candidate has no charged actual computation')
        bysource={source:[a for a in ref['attempts'] if a['source']==source] for source in
                  ('lbfgsb','de_0','de_polish_0','fine_lbfgsb','de_1','de_polish_1','powell')}
        for replicate in (0,1):
            if replicate==1 and not ref['full_bundle']:continue
            de=bysource['de_'+str(replicate)];polish=bysource['de_polish_'+str(replicate)]
            if (len(de)!=1 or len(polish)!=1 or de[0]['route']!='B' or polish[0]['route']!='B' or
                de[0]['seed']!=v1.de_seed(*canonical,0,replicate) or 'start' in de[0]):
                raise ValueError('Independent DE/polish evidence is missing or substituted')
        if not 1<=len(bysource['lbfgsb'])<=4 or len(bysource['fine_lbfgsb'])>8 or len(bysource['powell'])>2:
            raise ValueError('Reference start allowance mismatch')
        selector=ReferenceSearch.__new__(ReferenceSearch)
        for level in range(2 if ref['refined'] else 1):
            grid_source='fine_grid' if level else 'grid'
            attempt_source='fine_lbfgsb' if level else 'lbfgsb'
            grid=[Candidate(**c) for c in ref['candidates'] if c['source']==grid_source]
            count=8 if level and ref['full_bundle'] else 4
            starts=selector._starts(grid,Candidate(**prod),v1.grid_axes(v1.BY_NAME['B00'],row['n_beta'],level),count)
            observed=[tuple(a.get('start',())) for a in bysource[attempt_source]]
            if observed!=[c.point for c in starts]:
                raise ValueError('Required reference local starts did not occur')
        if ref['full_bundle']:
            distinct={}
            for c in sorted((Candidate(**c) for c in ref['candidates'] if c['source']!='powell'),
                            key=lambda c:(c.value,c.u,c.beta)):
                distinct.setdefault(c.point,c)
            starts=list(distinct)[:2]
            if [tuple(a.get('start',())) for a in bysource['powell']]!=starts:
                raise ValueError('Required Powell refinement attempts did not occur')
        elif bysource['de_1'] or bysource['de_polish_1'] or bysource['powell']:
            raise ValueError('Bundle attempts conflict with the recorded refinement state')
        for a in ref['attempts']:
            if a['bounds']!=[list(b) for b in v1.BY_NAME['B00'].bounds] or a['objective_calls']>a['objective_call_cap']:
                raise ValueError('Reference bounds or attempt cap mismatch')
            kind='de' if a['source'] in ('de_0','de_1') else 'powell' if a['source']=='powell' else 'local'
            cap=1024 if a['source']=='de_0' else 2032 if a['source']=='de_1' else 250
            if canonical_hash(a['settings'])!=canonical_hash(v1.numerical_settings()[kind]) or a['objective_call_cap']!=cap:
                raise ValueError('Reference optimizer settings changed')
            if a['objective_calls']<1 or a.get('interrupted')=='BudgetExhausted':
                raise ValueError('Required reference attempt was not completed')
        for source,attempts in bysource.items():
            observations=[c for c in ref['candidates'] if c['source']==source]
            cursor=0
            for a in attempts:
                count=a['objective_calls']+int(a.get('interrupted') is None)
                observed=observations[cursor:cursor+count];cursor+=count
                if (len(observed)!=count or any(c['route']!=a['route'] or c['seed']!=a['seed'] for c in observed)
                    or a.get('start') is not None and (observed[0]['u'],observed[0]['beta'])!=tuple(a['start'])):
                    raise ValueError('Reference attempt lacks its actual candidate observations')
            if cursor!=len(observations):
                raise ValueError('Reference candidate observations disagree with attempt counts')
        for level in range(2 if ref['refined'] else 1):
            traces={tuple(t['key']):t for t in row.get('trace_cache',[])}
            axes=v1.grid_axes(v1.BY_NAME['B11'],row['n_beta'],level)
            for axis,fixed in ((0,0.),(0,2*np.pi),(1,0.),(1,np.pi)):
                trace=traces.get(('B11',level,axis,fixed))
                if trace is None or trace.get('done') is not True:
                    raise ValueError('Required boundary trace is unfinished')
                coordinates={(c['u'],c['beta']) for c in trace['candidates'] if c['source']=='face_grid'}
                expected={(fixed,x) if axis==0 else (x,fixed) for x in axes[1-axis]}
                if coordinates!=expected:raise ValueError('Boundary node coverage mismatch')
                scalar={pack('>dd',c['u'],c['beta']) for c in trace['candidates'] if c['source']=='face_scalar'}
                if len(scalar)>150:raise ValueError('Boundary allowance exceeded')
        # Every grid node must be actually present as its declared route A source.
        for level in range(2 if ref['refined'] else 1):
            source='fine_grid' if level else 'grid'
            observed={(c['u'],c['beta']) for c in ref['candidates'] if c['source']==source and c['route']=='A'}
            required=set(product(*v1.grid_axes(v1.BY_NAME['B00'],row['n_beta'],level)))
            if observed!=required:raise ValueError('Reference grid coverage is incomplete')
    status,checks,pairs=assess(rows,keys,stopped=bool(s['stop_reason']))
    if status!='QUALIFIED':raise ValueError('Persisted core does not independently qualify')
    if m.get('status')!='QUALIFIED' or m.get('phase_status')!='QUALIFIED':
        raise ValueError('Core status conflicts with independently calculated qualification')
    with (directory/'paired_contrasts.csv').open() as f:persisted=list(csv.DictReader(f))
    if len(persisted)!=len(pairs):raise ValueError('Paired comparison coverage mismatch')
    pairmap={(int(p['instance_id']),int(p['initialization_id']),int(p['i']),int(p['j'])):p for p in persisted}
    if set(pairmap)!=set(expected_pairs(keys)):raise ValueError('Paired identities mismatch')
    for p in pairs:
        got=pairmap[p['instance_id'],p['initialization_id'],p['i'],p['j']]
        if int(got['policy'])!=33 or got['status']!=p['status'] or any(float(got[n])!=p[n] for n in ('K','production_contrast','reference_contrast')):
            raise ValueError('Paired values were not calculated by approved definitions')
    return actual


def create_lock(path,directory,resource,*,approval,publication_origin=None,_provisional=False):
    publication_origin=process_time() if publication_origin is None else publication_origin
    from ..calibration.resources import ResourceGuard
    guard=ResourceGuard(directory,path)
    guard.check(force=True,extra_memory=4*(Path(directory)/'budget.json').stat().st_size)
    driver=identity();hashes=verify_evidence(directory,expected_driver=driver)
    snapshot=current_snapshot(directory,resource,_allow_unsettled=_provisional)
    b=Budget.from_snapshot(snapshot,phase='validation')
    b.cpu_start=publication_origin;b.resource_guard=guard
    b.check_time();guard.check(b,force=True)
    if snapshot['limits']!=asdict(Limits()) or snapshot['phases']['development']>4_500_000:
        raise ValueError('Official lock requires the exact approved limits')
    p=dict(schema_version=2,rules_version=RULES,scope=SCOPE,protocol_sha256=PROTOCOL_HASH,
        approval_identity=approval,driver=driver,runtime=runtime_identity(),policy=33,domain='B00',
        reserves_active=True,thresholds=THRESHOLDS,manifest=manifest(),history=HISTORY,
        development_evidence=dict(directory=str(Path(directory).absolute()),sha256=hashes),
        resource_path=str(Path(resource).absolute()),resource_sha256=file_hash(resource),budget=snapshot,
        development_status='QUALIFIED')
    core_manifest=json.loads((Path(directory)/'calibration_manifest.json').read_text())
    p['diagnostics_authorized_before_validation']=core_manifest.get('authorization',{}).get('diagnostics_authorized_before_validation') is True
    if not approval:raise ValueError('Scientific approval identity is required')
    preflight_validation_lock(path)
    if Path(path).resolve().is_relative_to(Path(directory).resolve()):raise ValueError('Lock must be outside core evidence')
    b.check_time()
    atomic_json(path,dict(payload=p,sha256=canonical_hash(p)))
    # Bind publication/verification cost monotonically without changing the lock.
    overhead=process_time()-publication_origin
    cost=dict(schema_version=2,lock_sha256=file_hash(path),resource_sha256=file_hash(resource),
        resource_path=str(Path(resource).absolute()),
        cpu_seconds=snapshot['cpu_seconds']+overhead,cpu_measurement_cutoff=process_time(),
        process_ended=False,stop_reason=None)
    if cost['cpu_seconds']>=28_800:cost['stop_reason']='cpu_ceiling'
    atomic_json(str(path)+'.publication.json',dict(payload=cost,sha256=canonical_hash(cost)))
    return Path(path)


class ValidationGate:
    def __init__(self,path):
        if path is None:raise PermissionError('A v2 lock is required')
        self.path=Path(path);self.bytes_hash=file_hash(path)
        document=json.loads(self.path.read_text());p=document['payload']
        if (document['sha256']!=canonical_hash(p) or p.get('schema_version')!=2 or p.get('rules_version')!=RULES
            or p.get('protocol_sha256')!=PROTOCOL_HASH or p.get('scope')!=SCOPE or p.get('policy')!=33
            or p.get('domain')!='B00' or p.get('thresholds')!=THRESHOLDS or p.get('driver')!=identity()
            or p.get('history')!=HISTORY or canonical_hash(p.get('manifest'))!=canonical_hash(manifest())
            or p.get('runtime')!=runtime_identity() or not p.get('approval_identity')):
            raise ValueError('Lock is not the exact approved v2 capability')
        require_official_runtime()
        self.payload=p;self.payload_hash=canonical_hash(p);self.resource_path=Path(p['resource_path'])
        evidence=p['development_evidence']
        verify_evidence(evidence['directory'],hashes=evidence['sha256'],expected_driver=p['driver'])
        if file_hash(self.resource_path)!=p['resource_sha256']:raise ValueError('Resource identity changed')
        current=self.current_budget()
        Budget.from_snapshot(current,phase='validation')
        self.started=False;self.generation_claimed=False

    def current_budget(self):
        p=self.payload;s=current_snapshot(p['development_evidence']['directory'],self.resource_path)
        if any(canonical_hash(s[k])!=canonical_hash(p['budget'][k]) for k in s
               if k not in ('cpu_seconds','stop_reason')) or s['cpu_seconds']<p['budget']['cpu_seconds']:
            raise ValueError('Lock budget grant differs from development')
        pub=_record(str(self.path)+'.publication.json')
        if (pub.get('lock_sha256')!=self.bytes_hash or pub.get('resource_sha256')!=p['resource_sha256']
            or pub['cpu_seconds']<p['budget']['cpu_seconds']):
            raise ValueError('Publication overhead is unaccounted')
        settled=_record(str(self.resource_path)+'.settled.json')
        if settled.get('publication_sha256')==file_hash(str(self.path)+'.publication.json'):
            if s['cpu_seconds']<pub['cpu_seconds']:raise ValueError('Publication CPU is not settled')
        else:
            ended=_record(str(self.path)+'.publication-ended.json')
            if (not ended.get('process_ended') or ended['previous_sha256']!=file_hash(str(self.path)+'.publication.json')
                or ended['resource_settlement_sha256']!=file_hash(str(self.resource_path)+'.settled.json')
                or ended['cpu_seconds']<max(s['cpu_seconds'],pub['cpu_seconds'])):
                raise ValueError('Publisher sealing/finalization is not settled')
            s.update(cpu_seconds=ended['cpu_seconds'],stop_reason=ended['stop_reason'])
        if s['phase']!='development':raise ValueError('A validation capability requires development expenditure')
        return s

    @property
    def grant_path(self):
        settled=_record(str(self.resource_path)+'.settled.json')
        if settled.get('publication_sha256')==file_hash(str(self.path)+'.publication.json'):
            return self.resource_path
        return Path(str(self.path)+'.publication-ended.json')

    def authorize(self,policy):
        if policy!=33 or file_hash(self.path)!=self.bytes_hash or canonical_hash(self.payload)!=self.payload_hash:
            raise ValueError('Policy or immutable lock was substituted')
        return self.payload

    def begin_generation(self,policy):
        self.authorize(policy)
        if self.generation_claimed:raise ValueError('Validation generation is permitted once')
        evidence=self.payload['development_evidence']
        verify_evidence(evidence['directory'],hashes=evidence['sha256'],expected_driver=self.payload['driver'])
        Budget.from_snapshot(self.current_budget(),phase='validation')
        marker=Path(str(self.resource_path)+'.validation-consumed.json')
        atomic_json(marker,dict(lock_sha256=self.bytes_hash,protocol_sha256=PROTOCOL_HASH))
        self.generation_claimed=True;self.started=True
