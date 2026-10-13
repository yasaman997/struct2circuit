"""Frozen metadata, seed domains and identities; no instance generation."""
from hashlib import sha256
from pathlib import Path
import csv
import json
import os
import platform
import subprocess

import numpy as np
import scipy
from threadpoolctl import threadpool_info

from ..calibration.config import FAMILIES, PROTOCOL_HASH as V1_HASH
from ..calibration.storage import canonical_hash, driver_identity
from ..calibration_v2.config import PROTOCOL_HASH as V2_HASH

DESIGN_HASH = '032d4d997c5aefcde3bdef8ae64a206b744701073bf46f1974bf68793d0cac41'
BASE = '2042e4eb60d5eae71b4fbc1500af4ae065adde1f'
SCOPE = 'stage7o_alignment_c6_v1'
ROOT = Path(__file__).resolve().parents[3]
DESIGN = ROOT/'paper/controlled_mixer_alignment_v1.md'
ROLES = ('qubo', 'de0', 'de1', 'controls', 'order')
HISTORY_HASH = 'd7043055769685c8ed1858cb2cd4da6d6ba81372c703a1ca8126109007a578c5'
DEVELOPMENT = {
    'stage7b-development-e8dea4d-v1': '351ae59c04ad9cb4219b283ac440af65909c004ec21bad3928e9de72057da103',
    'stage7k-development-2042e4e-v2-p33-v1': 'be56f03d4e4e8f87fc6fc2ad6a8d386ca3af790689d6dbba4baeb04d5f38de13',
}


def seed(role, family, replicate):
    if role not in ROLES or type(family) is not int or type(replicate) is not int or not (0 <= family < 4 and 0 <= replicate < 8):
        raise ValueError('Unknown frozen role or panel identity')
    value = f'struct2circuit/stage7o/alignment-c6/v1|{role}|f={family}|r={replicate}'
    return int.from_bytes(sha256(value.encode()).digest(), 'big')


def panel():
    rows = [dict(family=f, replicate=r, n=6, k=3, generator=FAMILIES[f][0],
                 parameters=FAMILIES[f][1], seeds={role: seed(role, f, r) for role in ROLES},
                 arm_order=['high', 'low'] if seed('order', f, r) & 1 == 0 else ['low', 'high'])
            for r in range(8) for f in range(4)]
    seeds = [v for row in rows for v in row['seeds'].values()]
    if len(set(seeds)) != 160:
        raise ValueError('Role seed collision')
    return rows


def manifest():
    return dict(schema=1, scope=SCOPE, design_sha256=DESIGN_HASH, panel=panel(),
                n=6, k=3, depth=1, initialization='uniform', domain='B00',
                bounds=[[0., 2*np.pi], [0., np.pi]], independent_units=32, arms=64,
                limits=dict(arm=20000, total=1280000, numerical_cpu=3300., cpu=3600.,
                            wall=7200., rss=4*1024**3, launch_disk=5*1024**3, disk=1024**3),
                status='NOT_EXECUTED', independent_verification_required=True)


def public_seed_check(root=ROOT):
    """Audit the new role seeds against an explicit public metadata allowlist."""
    from ..benchmark import _derived_seed, _baseline_seed
    used, sources = set(), {}
    for name in ('configs/benchmark_v1_draft.json', 'results/pilot_results.csv',
                 'results/independent_dks_benchmark_v1_instances.csv'):
        data = (root/name).read_bytes()
        sources[name] = sha256(data).hexdigest()
        if name.endswith('.json'):
            config = json.loads(data)
            for split, entries in config['splits'].items():
                for i, entry in enumerate(entries):
                    for r in range(entry['count']):
                        value = _derived_seed(config['master_seed'], split, entry['family'], i, r)
                        used.add(value)
                        used.update(_baseline_seed(value, b) for b in range(config['random_baseline_replicates']))
        else:
            rows = csv.DictReader(data.decode().splitlines())
            if 'problem_seed' not in rows.fieldnames:
                raise ValueError('Missing public seed identifiers')
            used.update(int(row['problem_seed']) for row in rows)
    planned = {v for s in panel() for v in s['seeds'].values()}
    if planned & used:
        raise ValueError('Fresh panel seed collision; no redraw permitted')
    return dict(planned_seeds=160, public_seeds=len(used), sources=sources, protected_records_accessed=False)


def identity():
    inherited = driver_identity()
    digest = sha256(inherited['sha256'].encode())
    paths = sorted(Path(__file__).parent.glob('*.py')) + [ROOT/'experiments/run_controlled_alignment.py', DESIGN]
    for p in paths:
        digest.update(str(p.relative_to(ROOT)).encode()); digest.update(p.read_bytes())
    return dict(revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                sha256=digest.hexdigest(), inherited_sha256=inherited['sha256'],
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)


def verify_design():
    for p, expected in [(DESIGN, DESIGN_HASH), (ROOT/'paper/numerical_calibration_protocol_v1.md', V1_HASH),
                        (ROOT/'paper/numerical_calibration_protocol_v2.md', V2_HASH)]:
        if sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen scientific document changed: {p}')


def runtime():
    verify_design()
    software = identity()
    if tuple(software[k] for k in ('python', 'numpy', 'scipy')) != ('3.12.14', '2.5.3', '1.18.1'):
        raise ValueError('Execution requires Python 3.12.14 / NumPy 2.5.3 / SciPy 1.18.1')
    if any(os.environ.get(n) != '1' for n in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('Set all three numerical thread variables to 1 before imports')
    pools = threadpool_info()
    if not pools or any(p['num_threads'] != 1 for p in pools):
        raise ValueError('Loaded numerical pools must have one thread')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Official execution requires a clean committed implementation')
    inherited = ['src/struct2circuit/'+name for name in ('calibration', 'calibration_v2', 'mixers.py',
                  'problems.py', 'simulator.py', 'optimize.py')]
    if subprocess.check_output(['git', 'diff', BASE, '--', *inherited], cwd=ROOT, text=True).strip():
        raise ValueError('Inherited scientific implementation differs from approved main')
    return dict(software=software, pools=pools)
