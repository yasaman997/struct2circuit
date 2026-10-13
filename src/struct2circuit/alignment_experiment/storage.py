"""Immutable evidence, one-use authorization and read-only scalar verification."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import os
from struct import unpack

import numpy as np

from ..calibration.storage import canonical_hash
from ..calibration.config import BY_NAME
from ..calibration_v2.storage import atomic_json
from ..outputs import preflight_outputs
from ..problems import CardinalityQUBO
from .config import DESIGN_HASH, DEVELOPMENT, HISTORY_HASH, ROOT, SCOPE, identity, manifest, panel, public_seed_check, verify_design
from .evidence import analyze, bindings
from .selection import select
from .provenance import read_results, reconcile


def file_hash(path):
    p = Path(path)
    if p.is_symlink() or any(a.is_symlink() for a in p.parents):
        raise ValueError('Evidence and authorization paths may not use symlink aliases')
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fresh_output(path):
    p = Path(path).absolute()
    preflight_outputs((p, Path(str(p)+'.settled.json')))
    if any(a.is_symlink() for a in p.parents):
        raise ValueError('Output parent symlink aliases are forbidden')
    protected = [ROOT] + [Path('/workspace')/name for name in ('calibration-runs', 'calibration-run-provenance',
                    'calibration-archives', 'calibration-authorizations', 'research-archives')]
    resolved = p.resolve()
    if any(resolved == q.resolve() or q.resolve() in resolved.parents for q in protected):
        raise ValueError('Choose a fresh research output outside historical artifacts and the repository')
    return p


def check_authorization(a, output):
    """Pure capability/schema check shared by admission and later verification."""
    expected = dict(execution_authorized=True, scope=SCOPE, design_sha256=DESIGN_HASH,
                    implementation=identity(), manifest_sha256=canonical_hash(manifest()), output=str(Path(output).absolute()))
    if any(a.get(k) != v for k, v in expected.items()) or not a.get('approval_identity'):
        raise PermissionError('Authorization must bind this design, implementation, panel, scope and fresh destination')
    if set(a) != set(expected) | {'approval_identity', 'historical_archive', 'historical_archive_sha256', 'development_records'}:
        raise PermissionError('Unexpected authorization fields; validation and calibration phases are unavailable')
    archive = Path(a['historical_archive'])
    if archive != Path('/workspace/calibration-archives/stage7b-development-e8dea4d-v1.tar.gz') or a['historical_archive_sha256'] != HISTORY_HASH:
        raise PermissionError('Historical FAILED archive binding mismatch')
    records = a['development_records']
    if records != {str(Path('/workspace/calibration-runs')/name/'instances.jsonl'): digest for name, digest in DEVELOPMENT.items()}:
        raise PermissionError('Only the two preserved development identity records may be read')


def authorization(path, output):
    if path is None:
        raise PermissionError('Separate persisted mechanism execution authorization required')
    p = Path(path).absolute()
    file_hash(p)
    a = json.loads(p.read_text())
    output = fresh_output(output)
    check_authorization(a, output)
    if file_hash(a['historical_archive']) != HISTORY_HASH:
        raise PermissionError('Historical FAILED archive changed')
    records = a['development_records']
    excluded = set()
    for filename, digest in records.items():
        if file_hash(filename) != digest:
            raise PermissionError('Historical development identity record changed')
        for line in Path(filename).read_text().splitlines():
            row = json.loads(line)
            if row['instance_id'] not in range(8) or row['split'] == 'validation':
                raise PermissionError('Historical binding contains non-development data')
            if row.get('coefficients_sha256'):
                excluded.add(row['coefficients_sha256'])
    verify_design()
    # Public metadata only; this helper explicitly excludes protected manifests.
    public_seed_check()
    if Path(str(p)+'.used').exists() or Path(str(p)+'.used').is_symlink():
        raise PermissionError('Authorization was already consumed; no restart is permitted')
    return a, excluded


def claim(path, output, auth):
    """One-use marker is durable before generation; failures never restore it."""
    atomic_json(Path(str(Path(path).absolute())+'.used'), dict(authorization_sha256=file_hash(path),
                approval_identity=auth['approval_identity'], output=str(output)))
    output.mkdir(parents=True, exist_ok=False)
    with (output/'authorization.json').open('xb') as f:
        f.write(Path(path).read_bytes()); f.flush(); os.fsync(f.fileno())


def checkpoint(output, budget, name, *, previous=None):
    payload = dict(snapshot=budget.snapshot(), journal_sha256=file_hash(budget.path),
                   results_sha256=file_hash(budget.results_path),
                   previous=previous, previous_sha256=file_hash(output/previous) if previous else None)
    filename = f'resources-{name}.json'
    atomic_json(output/filename, dict(payload=payload, sha256=canonical_hash(payload)))
    return filename


def finish(output, budget, records, implementation, authorization_sha256, *, previous):
    atomic_json(output/'records.json', records)
    records = json.loads((output/'records.json').read_text())
    result = analyze(records)
    # A scalar summary is provisional; only a later independent verifier may
    # attest consistency. Neither runner nor verifier publishes a scientific claim.
    atomic_json(output/'provisional-analysis.json', result)
    latest = checkpoint(output, budget, 'final', previous=previous)
    files = {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}
    atomic_json(output/'inventory.json', dict(schema=1, scope=SCOPE, design_sha256=DESIGN_HASH,
        implementation=implementation, authorization_sha256=authorization_sha256, manifest=manifest(),
        files=files, latest_resource=latest, status='AWAITING_INDEPENDENT_VERIFICATION'))


def settle(output, cpu_seconds):
    inv = output/'inventory.json'
    # Measured after the evidence-writing worker exits, includes its imports,
    # preparations, numerical work, fsyncs and finalization, as in v2 wait4.
    atomic_json(Path(str(output)+'.settled.json'), dict(inventory_sha256=file_hash(inv), cpu_seconds=cpu_seconds,
                worker_receipt_sha256=file_hash(output/'worker-ended.json'),
                process_ended=True, cpu_accounting='wait4-exited-worker-and-waited-children',
                stop_reason='cpu_ceiling' if cpu_seconds >= 3600 else None))


def verify_ledger(output, latest):
    events = []
    previous = None
    with (output/'evaluations.jsonl').open() as f:
        for i, line in enumerate(f, 1):
            record = json.loads(line); e = record['payload']
            if (canonical_hash(e) != record['sha256'] or e['previous'] != previous or e['sequence'] != i
                or e['domain'] != 'B00'):
                raise ValueError('Evaluation journal chain mismatch')
            if not BY_NAME['B00'].contains(unpack('>dd', bytes.fromhex(e['coordinates']))):
                raise ValueError('Admitted computation lies outside B00')
            events.append(e); previous = record['sha256']
    chain, seen = [], set()
    while latest:
        if latest in seen or Path(latest).name != latest:
            raise ValueError('Resource chain alias/cycle')
        seen.add(latest)
        record = json.loads((output/latest).read_text()); p = record['payload']
        if canonical_hash(p) != record['sha256']:
            raise ValueError('Resource receipt digest mismatch')
        if p['previous'] and (Path(p['previous']).name != p['previous'] or file_hash(output/p['previous']) != p['previous_sha256']):
            raise ValueError('Linked resource receipt altered')
        chain.append(p); latest = p['previous']
    prior = None
    provenance = read_results(output/'evaluations.results.jsonl', events)
    result_records = provenance['records']
    for receipt in reversed(chain):
        snap = receipt['snapshot']; count = snap['counters'].get('actual_evaluations', 0)
        if (not np.isfinite(snap['cpu_seconds']) or snap['cpu_seconds'] < 0
            or not np.isfinite(snap['wall_seconds']) or snap['wall_seconds'] < 0
            or not (0 < snap['limits']['total'] <= 1280000 and 0 < snap['limits']['arm'] <= 20000
                    and 0 < snap['limits']['numerical_cpu'] <= 3300 < snap['limits']['cpu'] <= 3600)):
            raise ValueError('Invalid CPU/wall counters or inflated resource limits')
        if type(count) is not int or not 0 <= count <= len(events):
            raise ValueError('Invalid cumulative evaluation count')
        prefix = events[:count]
        result_count = snap.get('result_count')
        if type(result_count) is not int or not 0 <= result_count <= len(result_records):
            raise ValueError('Invalid numerical result/request cutoff')
        result_prefix = result_records[:result_count]
        if snap.get('result_tail') != (result_prefix[-1]['sha256'] if result_prefix else None):
            raise ValueError('Numerical result/request cutoff digest mismatch')
        requests = [r['payload'] for r in result_prefix if r['payload']['kind']=='request']
        cached = [r['payload'] for r in result_prefix if r['payload']['kind']=='resolution' and r['payload']['cached']]
        results = [r['payload'] for r in result_prefix if r['payload']['kind']=='result']
        prefix_ids = {canonical_hash(e) for e in prefix}
        if any(r['computation_id'] not in prefix_ids for r in results):
            raise ValueError('Resource cutoff contains results for uncharged computations')
        tail = canonical_hash(prefix[-1]) if prefix else None
        arms = dict(Counter(e['landscape'] for e in prefix))
        domains = [dict(arm=a, domain='B00', count=n) for a, n in sorted(arms.items())]
        if (snap['journal_tail'] != tail or snap['per_arm'] != arms or snap['per_domain'] != domains
            or snap['counters'].get('essential', 0) != count
            or snap['counters'].get('oracle_calls', 0) != sum(e['oracle'] for e in prefix)
            or snap['counters'].get('derivative_evolutions', 0) != sum(e['derivative'] for e in prefix)
            or snap['counters'].get('objective_requests', 0) != len(requests)
            or snap['counters'].get('cache_hits', 0) != len(cached)
            or snap['counters'].get('gradient_computations', 0) != sum(r['gradient_started'] for r in results)
            or snap['counters'].get('matrix_exponentials', 0) != sum(r['matrix_exponentials'] for r in results)
            or count > snap['limits']['total'] or any(n > snap['limits']['arm'] for n in arms.values())):
            raise ValueError('Resource counters do not match persisted journal')
        if prior and (count < prior['counters'].get('actual_evaluations', 0) or snap['cpu_seconds'] < prior['cpu_seconds']
                      or result_count < prior['result_count']
                      or snap['wall_seconds'] < prior['wall_seconds']
                      or any(snap['counters'].get(k, 0) < v for k, v in prior['counters'].items())):
            raise ValueError('Cumulative resource counters were reset')
        prior = snap
    if count != len(events) or chain[0]['journal_sha256'] != file_hash(output/'evaluations.jsonl'):
        raise ValueError('Final receipt omits numerical work')
    if result_count != len(result_records) or chain[0]['results_sha256'] != file_hash(output/'evaluations.results.jsonl'):
        raise ValueError('Final receipt omits numerical results/requests')
    return prior


def verify(output):
    """Read-only verification of coefficients, graph selection, scalar evidence and receipts."""
    output = Path(output).absolute()
    inventory = json.loads((output/'inventory.json').read_text())
    if canonical_hash(inventory['manifest']) != canonical_hash(manifest()) or inventory['design_sha256'] != DESIGN_HASH or inventory['implementation'] != identity():
        raise ValueError('Evidence design/source identity mismatch')
    for filename, digest in inventory['files'].items():
        if Path(filename).name != filename or file_hash(output/filename) != digest:
            raise ValueError('Artifact tampering or path alias')
    auth = json.loads((output/'authorization.json').read_text())
    check_authorization(auth, output)
    if (file_hash(output/'authorization.json') != inventory['authorization_sha256']
        or auth.get('implementation') != inventory['implementation'] or auth.get('design_sha256') != DESIGN_HASH
        or auth.get('output') != str(output) or auth.get('scope') != SCOPE
        or auth.get('manifest_sha256') != canonical_hash(manifest()) or auth.get('execution_authorized') is not True):
        raise ValueError('Stored execution authorization binding mismatch')
    settled = json.loads(Path(str(output)+'.settled.json').read_text())
    if (settled['inventory_sha256'] != file_hash(output/'inventory.json') or settled['process_ended'] is not True
        or settled['worker_receipt_sha256'] != file_hash(output/'worker-ended.json')):
        raise ValueError('Missing/altered process-exit resource settlement')
    snap = verify_ledger(output, inventory['latest_resource'])
    if snap['limits'] != dict(total=1280000, arm=20000, numerical_cpu=3300., cpu=3600.):
        raise ValueError('Official evidence changed frozen resource ceilings')
    if settled['cpu_seconds'] < snap['cpu_seconds'] or not np.isfinite(settled['cpu_seconds']):
        raise ValueError('Process-exit CPU cannot erase accrued CPU')
    worker = json.loads((output/'worker-ended.json').read_text())
    if worker['cpu_seconds'] < snap['cpu_seconds'] or settled['cpu_seconds'] < worker['cpu_seconds']:
        raise ValueError('Receipt finalization CPU was omitted')
    records = json.loads((output/'records.json').read_text())
    admissions = [json.loads(line)['payload'] for line in (output/'evaluations.jsonl').read_text().splitlines()]
    reconcile(records, read_results(output/'evaluations.results.jsonl', admissions))
    coefficients = set()
    allowed_objectives = {}
    specs = {(s['family'],s['replicate']):s for s in panel()}
    for row in records:
        if canonical_hash(row.get('spec')) != canonical_hash(specs.get((row['family'],row['replicate']))):
            raise ValueError('Frozen instance seeds/generator specification substituted')
        if row.get('Q') is None:
            if any(a.get('reference') for a in row['arms'].values()):
                raise ValueError('Numerical evidence without its QUBO')
            continue
        problem = CardinalityQUBO(row['Q'], row['c'], 3)
        selected = select(problem)
        if canonical_hash(selected) != canonical_hash(row['selection']) or selected['coefficients_sha256'] in coefficients:
            raise ValueError('Selection provenance or duplicate QUBO mismatch')
        coefficients.add(selected['coefficients_sha256'])
        if not selected['selected'] and any(a.get('reference') or a.get('controls') for a in row['arms'].values()):
            raise ValueError('Numerical evidence for an untestable alignment contrast')
        for arm, graph in selected['selected'].items():
            actual = row['arms'][arm]
            if actual['bindings'] is None and not actual.get('reference') and not actual.get('controls') and actual.get('unresolved_reason'):
                continue
            expected = bindings(problem, graph['edges'], f"{row['family']}:{row['replicate']}:{arm}", inventory['implementation'])
            if row['arms'][arm]['bindings'] != expected:
                raise ValueError('QUBO/graph/state/source identity substitution')
            from .reference import interior
            from ..mixers import MixerSpec, xy_mixer_hamiltonian
            h = xy_mixer_hamiltonian(problem.feasible_basis(), MixerSpec('c6', 6, tuple(map(tuple, graph['edges'])), 'relabeling'))
            width = float(np.ptp(np.linalg.eigh(h)[0]))
            if actual.get('mixer_width') != width:
                raise ValueError('Native spectral width binding differs')
            if row['arms'][arm].get('interior_points') != interior(row['family'], row['replicate'], width).tolist():
                raise ValueError('Preseeded numerical control points were substituted')
            p = (np.arange(6)+1) % 6
            inverse = np.argsort(p)
            transported_problem = CardinalityQUBO(problem.Q[np.ix_(inverse, inverse)], problem.c[inverse], 3)
            transported_edges = tuple(tuple(sorted((int(p[i]), int(p[j])))) for i, j in graph['edges'])
            transported_identity = bindings(transported_problem, transported_edges, f"{row['family']}:{row['replicate']}:{arm}", inventory['implementation'])['objective_identity']
            allowed_objectives[f"{row['family']}:{row['replicate']}:{arm}"] = {expected['objective_identity'], transported_identity}
            for control in actual.get('controls', []):
                if control['kind'] == 'cyclic_transport' and control.get('transported_identity') != transported_identity:
                    raise ValueError('Transport control substituted its comparison objective')
    if any(e['identity'] not in allowed_objectives.get(e['landscape'],set()) for e in admissions):
        raise ValueError('Journal computation belongs to an unbound objective/arm')
    result = analyze(records)
    if result != json.loads((output/'provisional-analysis.json').read_text()):
        # JSON converts integer family-map keys into strings.
        if canonical_hash(result) != canonical_hash(json.loads((output/'provisional-analysis.json').read_text())):
            raise ValueError('Provisional decision differs from independent scalar reconstruction')
    if settled['cpu_seconds'] >= 3600 or snap['wall_seconds'] >= 7200:
        result['outcome'] = 'INCONCLUSIVE'
    return dict(verification='CONSISTENT', analysis=result, settled_cpu_seconds=settled['cpu_seconds'],
                independent_reviewer_approval_required=True, scientific_publication_authorized=False)
