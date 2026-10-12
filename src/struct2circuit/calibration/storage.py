"""Exclusive, fail-closed calibration artifacts and persisted validation locks."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import csv
import json
import math
import platform
import os
import io
from contextlib import redirect_stdout
import subprocess
import tempfile

import numpy as np
import scipy

from ..outputs import preflight_outputs
from .config import PROTOCOL_HASH, SCOPE, build_manifest, instance_specs
from .decisions import THRESHOLDS
from .budget import Budget

ARTIFACTS = ('calibration_manifest.json', 'instances.jsonl', 'graphs.jsonl', 'candidates.jsonl',
             'landscapes.jsonl', 'paired_contrasts.csv', 'budget.json', 'decision.md')
PAIR_FIELDS = ('instance_id', 'initialization_id', 'i', 'j', 'policy', 'production_contrast',
               'reference_contrast', 'K', 'status', 'domains', 'production_order', 'reference_order',
               'first_contrast', 'second_contrast', 'first_order', 'second_order')


def clean(value):
    if is_dataclass(value): return clean(asdict(value))
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)): return [clean(x) for x in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def canonical_json(value):
    return json.dumps(clean(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def canonical_chunks(value):
    """The canonical_json byte contract, without copying large containers."""
    if is_dataclass(value): value = asdict(value)
    if isinstance(value, np.ndarray): value = value.tolist()
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, dict):
        values = {str(k): v for k, v in value.items()}
        yield '{'
        for i, key in enumerate(sorted(values)):
            if i: yield ','
            yield json.dumps(key)
            yield ':'
            yield from canonical_chunks(values[key])
        yield '}'
    elif isinstance(value, (tuple, list, set)):
        yield '['
        for i, item in enumerate(value):
            if i: yield ','
            yield from canonical_chunks(item)
        yield ']'
    else:
        if isinstance(value, float) and not math.isfinite(value): value = None
        yield json.dumps(value, separators=(',', ':'), allow_nan=False)


def canonical_hash(value):
    digest = sha256()
    for chunk in canonical_chunks(value): digest.update(chunk.encode())
    return digest.hexdigest()


def write_canonical(handle, value):
    # Bounded batching avoids one write per scalar without building full JSON.
    chunks, size = [], 0
    for chunk in canonical_chunks(value):
        chunks.append(chunk)
        size += len(chunk)
        if size >= 65536:
            handle.write(''.join(chunks))
            chunks, size = [], 0
    handle.write(''.join(chunks)+'\n')


def driver_identity():
    root = Path(__file__).resolve().parent
    digest = sha256()
    sources = list(root.glob('*.py')) + [root.parent/name for name in ('optimize.py', 'simulator.py', 'mixers.py', 'problems.py', 'outputs.py', 'benchmark.py')]
    sources.append(root.parents[2]/'experiments'/'run_numerical_calibration.py')
    for path in sorted(sources):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unknown'
    return dict(sha256=digest.hexdigest(), revision=revision,
                software=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))


def provenance():
    stream = io.StringIO()
    with redirect_stdout(stream): np.show_config()
    return dict(driver=driver_identity(), protocol_sha256=PROTOCOL_HASH,
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                started_at=datetime.now(timezone.utc).isoformat(), blas_configuration=stream.getvalue(),
                threads={name: os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})


class OutputStore:
    def __init__(self, directory, *, historical_root=None):
        self.directory = Path(directory)
        historical = Path(historical_root) if historical_root is not None else Path(__file__).resolve().parents[3]/'results'
        if '..' in self.directory.parts:
            raise FileExistsError('Output directory may not contain parent traversal')
        if self.directory.resolve().is_relative_to(historical.resolve()):
            raise FileExistsError('Choose a fresh output directory outside historical results')
        paths = [self.directory/name for name in ARTIFACTS]
        preflight_outputs(paths)
        # Require a genuinely new directory, including aliases and dangling links.
        if self.directory.exists() or self.directory.is_symlink():
            raise FileExistsError('Calibration output directory must be fresh and unoccupied')
        for parent in self.directory.parents:
            if parent.is_symlink():
                raise FileExistsError('Calibration output path may not have symlinked parents')
        self.directory.mkdir(parents=True, exist_ok=False)
        self.handles = {}
        try:
            for name in ARTIFACTS:
                self.handles[name] = (self.directory/name).open('x', encoding='utf-8', newline='')
        except BaseException:
            self.close()
            raise
        self.csv_writer = csv.DictWriter(self.handles['paired_contrasts.csv'], fieldnames=PAIR_FIELDS)
        self.csv_writer.writeheader()
        self.handles['paired_contrasts.csv'].flush()

    def json(self, name, value, *, update_owned=False):
        handle = self.handles[name]
        if handle.tell():
            if not update_owned or name != 'calibration_manifest.json':
                raise ValueError('Only the currently owned manifest may be finalized')
            handle.seek(0)
            handle.truncate()
        write_canonical(handle, value)
        handle.flush()

    def row(self, name, value):
        handle = self.handles[name]
        handle.write(canonical_json(value)+'\n')
        handle.flush()

    def pair(self, value):
        value = clean(value)
        self.csv_writer.writerow({k: '' if v is None else v for k, v in value.items()})
        self.handles['paired_contrasts.csv'].flush()

    def decision(self, value):
        self.handles['decision.md'].write(value+'\n')
        self.handles['decision.md'].flush()

    def close(self):
        for handle in self.handles.values():
            handle.close()

    def sync(self):
        for handle in self.handles.values():
            handle.flush()
            os.fsync(handle.fileno())
        _sync_directory(self.directory)

    def __enter__(self): return self
    def __exit__(self, *args): self.close()



def preflight_validation_lock(path):
    path = Path(path)
    historical = Path(__file__).resolve().parents[3]/'results'
    if path.resolve().is_relative_to(historical.resolve()) or any(parent.is_symlink() for parent in path.parents):
        raise FileExistsError('Validation locks must be outside historical results and symlink aliases')
    if any(parent.exists() and not parent.is_dir() for parent in path.parents):
        raise FileExistsError('Validation lock parent must be a directory')
    preflight_outputs([path])
    return path


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _development_evidence(directory, payload, *, expected_hashes=None):
    """Check persisted completion independently of the producing process."""
    if directory is None:
        raise ValueError('A validation lock requires finalized development artifacts')
    directory = Path(directory)
    hashes = {}
    for name in ARTIFACTS:
        path = directory/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing or incomplete development artifact: {name}')
        with path.open('rb') as handle:
            hashes[name] = sha256_file(handle)
    if expected_hashes is not None and hashes != expected_hashes:
        raise ValueError('Finalized development evidence was altered')
    manifest = json.loads((directory/'calibration_manifest.json').read_text())
    specs = tuple(s for s in instance_specs(payload['reserves_active']) if s.split != 'validation')
    expected = {(s.instance_id, g, init) for s in specs for g in range(8) for init in range(3)}
    if (manifest.get('phase') != 'development' or manifest.get('development_finalized') is not True
        or manifest.get('status') != 'INCONCLUSIVE' or manifest.get('phase_status') != payload['development_status']
        or manifest.get('selected_policy') != payload['policy'] or manifest.get('validation_executed') is not False
        or manifest.get('protocol_sha256') != payload['protocol_sha256']
        or manifest.get('provenance', {}).get('driver') != payload['driver']
        or manifest.get('reserves_active') != payload['reserves_active']
        or len(manifest.get('expected_phase_keys', ())) != len(expected)
        or set(map(tuple, manifest.get('expected_phase_keys', ()))) != expected
        or any(canonical_json(manifest.get(k)) != canonical_json(payload['manifest'][k])
               for k in ('instances', 'graph_labels', 'initializations', 'settings'))):
        raise ValueError('Development manifest is not finalized consistently with the lock')
    persisted = json.loads((directory/'budget.json').read_text())
    if (not persisted.get('mandatory_complete') or persisted.get('stop_reason')
        or persisted.get('incomplete_required_work') != []
        or any(canonical_hash(persisted.get(k)) != canonical_hash(payload['budget'][k])
               for k in ('limits', 'counters', 'per_landscape', 'per_domain', 'unique_keys'))):
        raise ValueError('Development budget or mandatory work is incomplete')
    Budget.from_snapshot(persisted)
    decision = (directory/'decision.md').read_text()
    if any(line not in decision.splitlines() for line in (
        f'Phase numerical status: {payload["development_status"]}.',
        f'Policy: P{payload["policy"]} on B00 only.',
        f'Scope: {SCOPE}. Validation completed: False. Synthetic check: False.')):
        raise ValueError('Development decision does not match the lock')
    for name in ('graphs.jsonl', 'landscapes.jsonl'):
        rows = [json.loads(line) for line in (directory/name).read_text().splitlines()]
        if len(rows) != len(expected) or {tuple(row.get('landscape', ())) for row in rows} != expected:
            raise ValueError(f'Development coverage is incomplete: {name}')
        if name == 'landscapes.jsonl' and any(row.get('status') != 'observed' or row.get('error')
            or row.get('arithmetic_failure') or row.get('pending_trigger_work')
            or any(not row.get('references', {}).get(box, {}).get('complete') for box in
                   ('B00', 'B10', 'B01', 'B11')+(('B22', 'Bsign') if row.get('required_challenges') else ())) for row in rows):
            raise ValueError('Development landscape evidence is unresolved')
    instances = [json.loads(line) for line in (directory/'instances.jsonl').read_text().splitlines()]
    if len(instances) != len(specs) or {row.get('instance_id') for row in instances if row.get('status') == 'generated'} != {s.instance_id for s in specs}:
        raise ValueError('Development instance evidence is incomplete')
    return dict(directory=str(directory.resolve()), sha256=hashes)


def sha256_file(handle):
    digest = sha256()
    for chunk in iter(lambda: handle.read(1024*1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def create_validation_lock(path, *, policy, scope, reserves, budget, development_status, driver=None,
                           protocol_hash=PROTOCOL_HASH, development_output=None):
    if policy not in (9, 17, 33) or scope != SCOPE or protocol_hash != PROTOCOL_HASH:
        raise ValueError('A lock requires an approved B00 policy, explicit scope, and protocol identity')
    if policy in (9, 33) and not reserves:
        raise ValueError('A modified policy requires activated development reserves')
    if development_status not in ('ACCEPTED', 'ACCEPTED WITH EXPLICIT LIMITATIONS') or not budget.mandatory_complete or budget.stop_reason:
        raise ValueError('Unresolved development cannot create a validation lock')
    manifest = build_manifest(reserves)
    payload = dict(policy=policy, domain='B00', scope=scope, protocol_sha256=protocol_hash,
                   driver=driver or driver_identity(), thresholds=THRESHOLDS,
                   manifest=manifest, budget=budget.snapshot(), reserves_active=reserves,
                   rules_version='numerical_calibration_protocol_v1', development_status=development_status)
    payload['development_evidence'] = _development_evidence(development_output, payload)
    budget.check_time()
    payload['budget'] = budget.snapshot()
    document = dict(payload=payload, sha256=canonical_hash(payload))
    path = preflight_validation_lock(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish a fully synced file atomically and exclusively. A crash while
    # writing its temporary file cannot expose a partially written usable lock.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name+'.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            write_canonical(handle, document)
            handle.flush()
            os.fsync(handle.fileno())
        # Streaming/hash/fsync work also consumes time and disk. Check again
        # before the atomic publication can expose a usable capability.
        if budget.resource_guard is not None:
            budget.resource_guard.check(budget, force=True)
        budget.check_time()
        os.link(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return ValidationGate(path, expected_driver=payload['driver'])


class ValidationGate:
    def __init__(self, path, *, expected_driver=None):
        self.path = Path(path)
        self.bytes = self.path.read_bytes()
        document = json.loads(self.bytes)
        payload = document['payload']
        if document['sha256'] != canonical_hash(payload):
            raise ValueError('Validation lock checksum mismatch')
        self._payload_hash = document['sha256']
        self._bytes_hash = sha256(self.bytes).hexdigest()
        # JSON repeats equal model identities. Share their immutable strings;
        # canonical bytes and key values remain unchanged.
        identities = {}
        for row in payload.get('budget', {}).get('unique_keys', ()):
            row[0] = identities.setdefault(row[0], row[0])
        expected_manifest = build_manifest(payload.get('reserves_active', False))
        if (payload.get('protocol_sha256') != PROTOCOL_HASH or payload.get('thresholds') != THRESHOLDS
            or payload.get('domain') != 'B00' or payload.get('policy') not in (9, 17, 33)
            or payload.get('scope') != SCOPE or not payload.get('budget', {}).get('mandatory_complete')
            or canonical_json(payload.get('manifest')) != canonical_json(expected_manifest)
            or payload.get('rules_version') != 'numerical_calibration_protocol_v1'
            or payload.get('development_status') not in ('ACCEPTED', 'ACCEPTED WITH EXPLICIT LIMITATIONS')
            or payload.get('policy') in (9, 33) and not payload.get('reserves_active')
            or payload.get('driver') != (expected_driver or driver_identity())):
            raise ValueError('Validation lock does not match approved driver/configuration')
        Budget.from_snapshot(payload['budget'])
        evidence = payload.get('development_evidence', {})
        _development_evidence(evidence.get('directory'), payload, expected_hashes=evidence.get('sha256', {}))
        self.payload, self.started, self.generation_claimed = payload, False, False

    def begin_generation(self, policy):
        evidence = self.payload['development_evidence']
        _development_evidence(evidence['directory'], self.payload, expected_hashes=evidence['sha256'])
        self.authorize(policy)
        if self.generation_claimed:
            raise ValueError('Validation generation is permitted once per locked run')
        self.generation_claimed = True

    def authorize(self, policy):
        with self.path.open('rb') as handle:
            file_hash = sha256_file(handle)
        if (file_hash != self._bytes_hash or sha256(self.bytes).hexdigest() != self._bytes_hash
            or canonical_hash(self.payload) != self._payload_hash):
            raise ValueError('Validation lock was altered')
        if policy != self.payload['policy']:
            raise ValueError('Production policy substitution prohibited')
        if not self.started:
            marker = self.path.with_name(self.path.name+'.validation-started')
            preflight_validation_lock(marker)
            with marker.open('x', encoding='utf-8') as handle:
                handle.write(sha256(self.bytes).hexdigest()+'\n')
            self.started = True
        return self.payload
