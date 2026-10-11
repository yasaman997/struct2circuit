"""Approved metadata and domains. Manifest construction generates no QUBOs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import ceil, pi
from pathlib import Path
import csv
import json

import numpy as np

from ..mixers import (exchange_profile_scores, random_connected_mixer, ring_mixer,
                      score_conditioned_mixer, structure_conditioned_mixer)
from .budget import Limits
from .decisions import THRESHOLDS

BASELINE = "15084cb2f7c8e09b0463a742b1a5cbddb718f1b4"
PROTOCOL_HASH = "6c9945bf4c6d5c3d8c2b71e6f80a78d6dd7b671f02a12c70aa2bf4c1a734e9a1"
INITIALIZATIONS = ("uniform", "mixer_low", "mixer_high")
GRAPH_LABELS = ("ring", "strong_abs_Q", "weak_abs_Q", "low_RMS", "high_RMS",
                "shuffled_low_RMS", "shuffled_high_RMS", "random_connected")
FAMILIES = (
    ("block_correlated_qubo", dict(n_blocks=2, block_strength=.72, cross_strength=.16, linear_scale=.55)),
    ("weighted_densest_k_subgraph_qubo", dict(density=.55, weight_range=(.5, 1.5), weight_distribution="uniform", planted_community_strength=1.8)),
    ("weighted_max_k_vertex_cover_qubo", dict(density=.55, weight_range=(.5, 1.5), weight_distribution="uniform", community_strength=1.5, hub_strength=1.5)),
    ("weak_structure_null_qubo", dict(density=.55, coefficient_scale=1.)),
)
AUDIT_GRAPHS = (0, 7, 3, 4)
SENTINELS = ((0, 0), (4, 2), (3, 1), (7, 0))
SCOPE = 'bounded_B00_native_common_physical_beta'


def numerical_settings():
    """Frozen recording contract; solver calls are independently tested against it."""
    return dict(thresholds=THRESHOLDS.copy(), budgets=asdict(Limits()),
        local=dict(method='L-BFGS-B', maxiter=120, ftol=1e-13, gtol=1e-9, objective_call_cap=250),
        de=dict(strategy='best1bin', popsize=8, routine_maxiter=63, refined_maxiter=126,
                tol=1e-8, atol=1e-10, mutation=(.5, 1.), recombination=.7,
                init='latinhypercube', updating='immediate', workers=1, polish=False),
        powell=dict(maxiter=180, ftol=1e-12, xtol=1e-10, objective_call_cap=250, starts=2),
        boundary=dict(xatol=1e-10, bracket_calls=60, fresh_values_per_trace_per_level=150,
                      boundary_fraction=1e-8, near_boundary_fraction=.01),
        grid=dict(B11_u_intervals=64, beta_intervals_rule='next_power_of_two(max(64,8*W_max))', finer_levels=1),
        oracles=dict(pointwise_tolerance=1e-10, gradient_tolerance=1e-6, h_u=1e-5,
                     h_beta='1e-5/max(1,W)', interior_count=8, interior_distribution='uniform B11 with finite-difference margins',
                     cyclic_permutation='np.roll(np.arange(n),1)'),
        escalation_triggers=('R', 'incomplete_faces', 'E', 'K', 'guard_band', 'near_nonflat_face', 'domain_gain', 'ranking', 'scheduled_audit'),
        scope=SCOPE, broader_adequacy_essential=False, optional_spectral_time='OMITTED',
        qualification='B00 only; empirical finite panel; validation mandatory')


@dataclass(frozen=True)
class Domain:
    name: str
    box_id: int
    u: tuple[float, float]
    beta: tuple[float, float]

    def __post_init__(self):
        if not np.isfinite((*self.u, *self.beta)).all() or any(a >= b for a, b in (self.u, self.beta)):
            raise ValueError("A domain needs finite, strictly increasing bounds")

    @property
    def bounds(self):
        return (self.u, self.beta)

    def contains(self, point):
        return len(point) == 2 and all(np.isfinite(x) and a <= x <= b for x, (a, b) in zip(point, self.bounds))

    def includes(self, other):
        return all(a <= c and d <= b for (a, b), (c, d) in zip(self.bounds, other.bounds))

    def face_distances(self, point):
        if not self.contains(point):
            raise ValueError("Point outside declared domain")
        return tuple((x-a)/(b-a) for x, (a, b) in zip(point, self.bounds)), tuple((b-x)/(b-a) for x, (a, b) in zip(point, self.bounds))


DOMAINS = tuple(Domain(name, i, u, beta) for i, (name, u, beta) in enumerate((
    ("B00", (0., 2*pi), (0., pi)), ("B10", (0., 4*pi), (0., pi)),
    ("B01", (0., 2*pi), (0., 2*pi)), ("B11", (0., 4*pi), (0., 2*pi)),
    ("B22", (0., 8*pi), (0., 4*pi)), ("Bsign", (0., 4*pi), (-2*pi, 2*pi)),
)))
BY_NAME = {box.name: box for box in DOMAINS}
PRIMARY = DOMAINS[:4]


def spectral_domain(ring_width, width):
    if not np.isfinite([ring_width, width]).all() or min(ring_width, width) <= 0:
        raise ValueError("Spectral widths must be positive and finite")
    return Domain("Bv", 6, (0., 2*pi), (0., pi*ring_width/width))


def beta_intervals(width_max):
    if not np.isfinite(width_max) or width_max <= 0:
        raise ValueError("Positive finite feasible spectral width required")
    return 1 << (ceil(max(64., 8*width_max))-1).bit_length()


def grid_axes(domain, n_beta, level=0):
    """Dyadic indexing preserves identical float64 old nodes across nested grids."""
    if n_beta < 64 or n_beta & (n_beta-1) or level not in (0, 1):
        raise ValueError("Use the approved dyadic beta mesh and at most one refinement")
    denominators = (16*2**level, (n_beta//2)*2**level)
    axes = []
    for (lo, hi), denominator in zip(domain.bounds, denominators):
        start, stop = ceil(lo/pi*denominator), int(np.floor(hi/pi*denominator))
        values = pi*(np.arange(start, stop+1, dtype=float)/denominator)
        values = values[(values >= lo) & (values <= hi)]
        axes.append(np.unique(np.concatenate(([lo], values, [hi]))))
    return tuple(axes)


@dataclass(frozen=True)
class InstanceSpec:
    instance_id: int
    family: int
    split: str
    n: int
    k: int
    seed: int


def instance_specs(reserves=False):
    dev = [InstanceSpec(2*f, f, "development", 6, 3, 420000000+1000*f) for f in range(4)]
    if reserves:
        dev += [InstanceSpec(2*f+1, f, "reserve", 6, 3, 420000001+1000*f) for f in range(4)]
    val = [InstanceSpec(8+f, f, "validation", 9, 4, 420100000+1000*f) for f in range(4)]
    return tuple(sorted(dev+val, key=lambda s: s.instance_id))


def de_seed(instance_id, graph_id, initialization_id, box_id, replicate=0):
    if not (instance_id >= 0 and 0 <= graph_id < 8 and 0 <= initialization_id < 3 and 0 <= box_id < 7 and replicate in (0, 1)):
        raise ValueError("Invalid search identity")
    namespace, offset = (720000000, box_id) if box_id < 4 else (730000000, box_id-4)
    return namespace+10000*instance_id+100*graph_id+10*initialization_id+2*offset+replicate


def verify_seed_assignments():
    seeds = [de_seed(i, g, s, b, r) for i in range(12) for g in range(8)
             for s in range(3) for b in range(7) for r in range(2)]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Colliding DE seed assignments")
    return len(seeds)


def verify_public_seed_metadata(root):
    """Audit only declared public configuration/seed-use metadata, never manifests
    containing protected records or generated QUBOs. Recheck before execution.
    """
    from ..benchmark import _derived_seed, _baseline_seed
    root = Path(root)
    sources = ('configs/benchmark_v1_draft.json', 'results/pilot_results.csv',
               'results/independent_dks_benchmark_v1_instances.csv')
    used, hashes = set(), {}
    for name in sources:
        path = root/name
        data = path.read_bytes()  # Explicit public allowlist, no blind manifest discovery.
        hashes[name] = sha256(data).hexdigest()
        if name.endswith('.json'):
            config = json.loads(data)
            for split, entries in config['splits'].items():
                for entry_id, entry in enumerate(entries):
                    for replicate in range(entry['count']):
                        seed = _derived_seed(config['master_seed'], split, entry['family'], entry_id, replicate)
                        used.add(seed)
                        used.update(_baseline_seed(seed, r) for r in range(config['random_baseline_replicates']))
        else:
            rows = csv.DictReader(data.decode().splitlines())
            if 'problem_seed' not in rows.fieldnames:
                raise ValueError(f'Public seed metadata unavailable in {name}; obtain an external identifier check')
            used.update(int(row['problem_seed']) for row in rows)
    planned = {de_seed(i, g, init, b, r) for i in range(12) for g in range(8) for init in range(3)
               for b in range(7) for r in (0, 1)}
    planned.update(s.seed for s in instance_specs(True))
    planned.update(base+i for base in (520000000, 620000000) for i in range(12))
    planned.update(820000000+10000*i+100*g+10*init for i in range(12) for g in range(8) for init in range(3))
    collision = planned & used
    if collision:
        raise ValueError(f'Calibration seed collision: {sorted(collision)}; no redraw or namespace revision authorized')
    return dict(status='public_metadata_noncollision', sources=hashes, planned_seeds=len(planned),
                public_seeds=len(used), protected_records_accessed=False,
                instance_identifier_namespace='numerical_calibration_protocol_v1')


def build_manifest(reserves=False):
    specs = instance_specs(reserves)
    verify_seed_assignments()
    landscapes = [(s.instance_id, g, init) for s in specs for g in range(8) for init in range(3)]
    audits = [(s.instance_id, AUDIT_GRAPHS[s.family], init) for s in specs for init in range(3)]
    sentinels = [(s.instance_id, *SENTINELS[s.family]) for s in specs]
    return dict(protocol_sha256=PROTOCOL_HASH, baseline=BASELINE, status="NOT_EXECUTED",
                instance_identifier_namespace='numerical_calibration_protocol_v1',
                benchmark_status="DRAFT_UNFROZEN", reserves_active=reserves,
                instances=[asdict(s) | dict(generator=FAMILIES[s.family][0], parameters=FAMILIES[s.family][1].copy(),
                    random_seed=520000000+s.instance_id, shuffle_seed=620000000+s.instance_id) for s in specs],
                graph_labels=GRAPH_LABELS, initializations=INITIALIZATIONS, landscapes=landscapes,
                primary_boxes=[asdict(b) for b in PRIMARY], challenge_boxes=[asdict(b) for b in DOMAINS[4:]],
                settings=numerical_settings(), audits=audits, sentinels=sentinels,
                search_seeds=[dict(landscape=key, requested_DE=[de_seed(*key, b, r) for b in range(7) for r in (0, 1)],
                                   interior=820000000+10000*key[0]+100*key[1]+10*key[2]) for key in landscapes],
                mandatory_challenges=[(*key, box) for key in sentinels for box in ("B22", "Bsign")])


def verify_protocol(path):
    data = Path(path).read_bytes()
    if sha256(data).hexdigest() != PROTOCOL_HASH:
        raise ValueError("Approved protocol hash mismatch; do not execute")
    return PROTOCOL_HASH


def construct_graphs(problem, instance_id):
    """Construct once from a supplied problem; never generates or redraws QUBOs."""
    n, m = problem.n, problem.n
    rms = exchange_profile_scores(problem.Q, problem.c, problem.k)
    upper = np.triu_indices(n, 1)
    permutation = np.random.default_rng(620000000+instance_id).permutation(len(upper[0]))
    shuffled = np.zeros_like(rms)
    shuffled[upper] = rms[upper][permutation]
    shuffled += shuffled.T
    graphs = (ring_mixer(n), structure_conditioned_mixer(problem.Q, m),
              score_conditioned_mixer(abs(problem.Q), m, prefer="low"),
              score_conditioned_mixer(rms, m, prefer="low"), score_conditioned_mixer(rms, m, prefer="high"),
              score_conditioned_mixer(shuffled, m, prefer="low"), score_conditioned_mixer(shuffled, m, prefer="high"),
              random_connected_mixer(n, m, seed=520000000+instance_id))
    canonical, seen = [], {}
    for i, graph in enumerate(graphs):
        canonical.append(seen.setdefault((graph.n, graph.edges), i))
    return graphs, tuple(canonical), permutation.tolist()
