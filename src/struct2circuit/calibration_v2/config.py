"""Metadata only: constructing this manifest never generates a QUBO."""
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy

from ..calibration import config as v1
from ..calibration.decisions import THRESHOLDS
from ..calibration.storage import canonical_hash, driver_identity as v1_identity

PROTOCOL_HASH = 'abcb7cb1b7fc7b2f2a114b6c5a4328acc18a78aebca00544a20c383bd719548d'
RULES = 'numerical_calibration_protocol_v2'
SCOPE = 'P33_B00_native_XY_finite_panel_v2'
BASELINE = 'e8dea4d09f3a40542b361a9e191d3dacf5e295f7'
HISTORY = dict(revision=BASELINE, protocol_sha256=v1.PROTOCOL_HASH, status='FAILED',
    archive_sha256='d7043055769685c8ed1858cb2cd4da6d6ba81372c703a1ca8126109007a578c5',
    ineligible_policies=[9, 17])


def specs(phase):
    if phase not in ('development', 'validation'):
        raise ValueError('Expected development or validation')
    return tuple(s for s in v1.instance_specs(True)
                 if (s.split == 'validation') == (phase == 'validation'))


def expected_keys(phase):
    return tuple((s.instance_id, g, init) for s in specs(phase)
                 for g in range(8) for init in range(3))


def expected_pairs(keys):
    strata = sorted({(i, init) for i, _, init in keys})
    return tuple((i, init, a, b) for i, init in strata
                 for a, b in combinations(sorted(g for j, g, s in keys if (j, s) == (i, init)), 2))


def manifest():
    from .budget import Limits
    settings = v1.numerical_settings()
    settings.update(budgets=asdict(Limits()), policies=[33], essential_domains=['B00'],
                    scope=SCOPE, qualification='P33/B00 finite-panel only',
                    broader_adequacy_essential=False)
    return dict(schema_version=2, rules_version=RULES, protocol_sha256=PROTOCOL_HASH,
        scope=SCOPE, policy=33, domain='B00', history=HISTORY, reserves_active=True,
        instances=[asdict(s) | dict(generator=v1.FAMILIES[s.family][0], parameters=v1.FAMILIES[s.family][1])
                   for s in v1.instance_specs(True)], graph_labels=v1.GRAPH_LABELS,
        initializations=v1.INITIALIZATIONS, settings=settings,
        development_keys=expected_keys('development'), validation_keys=expected_keys('validation'),
        development_pairs=expected_pairs(expected_keys('development')),
        validation_pairs=expected_pairs(expected_keys('validation')),
        audits=v1.AUDIT_GRAPHS, sentinels=v1.SENTINELS,
        diagnostic_domains=[asdict(v1.BY_NAME[n]) for n in ('B10', 'B01', 'B11', 'B22', 'Bsign')],
        status='NOT_EXECUTED', benchmark_status='DRAFT_UNFROZEN')


def verify_protocol(path):
    if sha256(Path(path).read_bytes()).hexdigest() != PROTOCOL_HASH:
        raise ValueError('Approved v2 protocol hash mismatch')


def identity():
    root = Path(__file__).resolve().parent
    inherited = v1_identity()
    digest = sha256(inherited['sha256'].encode())
    for p in sorted(root.glob('*.py')) + [root.parents[2]/'experiments/run_numerical_calibration_v2.py']:
        digest.update(p.name.encode()); digest.update(p.read_bytes())
    return dict(revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                sha256=digest.hexdigest(), v1_sources_sha256=inherited['sha256'],
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
