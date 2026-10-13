"""Explicit toy coefficients and synthetic scalar records; no official generation/search."""
from collections import Counter
from copy import deepcopy
from itertools import combinations, product
from math import pi
from pathlib import Path
from struct import pack
from types import SimpleNamespace
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from struct2circuit.problems import CardinalityQUBO
from struct2circuit.mixers import MixerSpec, xy_mixer_hamiltonian
from struct2circuit.calibration.config import BY_NAME, grid_axes, numerical_settings
from struct2circuit.calibration.objective import Objective as NativeObjective
from struct2circuit.calibration.reference import Candidate, ReferenceResult
from struct2circuit.calibration.storage import canonical_hash
from struct2circuit.calibration_v2.storage import atomic_json
from struct2circuit.alignment_experiment import config, selection, evidence, storage
from struct2circuit.alignment_experiment.budget import Budget, ArmExhausted, BudgetExhausted, Guard
from struct2circuit.alignment_experiment.reference import Objective, ReferenceSearch


def toy(symmetric=False):
    q = np.zeros((6, 6))
    q[0, 2] = q[2, 0] = .125
    q[1, 4] = q[4, 1] = -.375
    c = np.array([.25, -.75, 1.125, .5, -1.25, 2.])
    if symmetric:
        q[:] = 0.; c[:] = .5
    return CardinalityQUBO(q, c, 3, 'unrelated-explicit-toy')


def scalar_arm(value=.2):
    """Complete synthetically flat scalar landscape, never a quantum computation."""
    bound = dict(objective_identity='toy-identity', coefficients_sha256='toy', graph_sha256='toy',
                 state_sha256='toy', domain='B00', design_sha256=config.DESIGN_HASH, implementation='toy')
    candidates = []
    attempts = []
    faces = []
    settings = numerical_settings()
    for level, source, n in [(0, 'grid', 4), (1, 'fine_grid', 8)]:
        axes = grid_axes(BY_NAME['B00'], 128, level)
        for u, beta in product(*axes):
            candidates.append(dict(u=float(u), beta=float(beta), value=value, route='A', source=source,
                                   objective_identity='toy-identity'))
        for j in range(n):
            start = [0., float(axes[1][2*j])]
            attempts.append(dict(source='lbfgsb' if level == 0 else 'fine_lbfgsb', route='A', start=start,
                seed=None, bounds=[list(x) for x in BY_NAME['B00'].bounds], settings=settings['local'],
                objective_call_cap=250, objective_calls=5, interrupted=None))
        for axis in (0, 1):
            for fixed in BY_NAME['B00'].bounds[axis]:
                points = [[fixed, float(x)] if axis == 0 else [float(x), fixed] for x in axes[1-axis]]
                faces.append(dict(level=level, axis=axis, fixed=fixed, complete=True,
                                  flat=axis == 1 and fixed == 0, points=points, values=[value]*len(points), brackets=[]))
    for r in (0, 1):
        for source, kind, cap in [(f'de_{r}', 'de', 1024 if r == 0 else 2032), (f'de_polish_{r}', 'local', 250)]:
            attempts.append(dict(source=source, route='B', seed=config.seed(f'de{r}', 0, 0),
                bounds=[list(x) for x in BY_NAME['B00'].bounds], settings=settings[kind],
                objective_call_cap=cap, objective_calls=5, interrupted=None))
        candidates.append(dict(u=.73, beta=.33, value=value, route='B', source=f'de_{r}',
                               seed=config.seed(f'de{r}',0,0), objective_identity='toy-identity'))
    for start in [[0., 0.], [0., pi/128]]:
        attempts.append(dict(source='powell', route='A', start=start, seed=None,
            bounds=[list(x) for x in BY_NAME['B00'].bounds], settings=settings['powell'],
            objective_call_cap=250, objective_calls=5, interrupted=None))
    points = [[.11+j*.01, .22+j*.01] for j in range(8)]
    controls = []
    def control(kind, point):
        row = dict(kind=kind, point=list(point), domain='B00', status='pass', bindings=bound)
        if kind == 'direct_expm':
            row['errors'] = {k: 0. for k in ('probability', 'gap', 'normalization', 'native_normalization', 'native_state_norm')}
        else:
            row['error'] = 0.
        if kind == 'gradient':
            row.update(analytic=[0., 0.], numerical=[0., 0.], steps=[1e-5,1e-5/8])
        if kind == 'candidate_reevaluation':
            row.update(value=value, checked_value=value)
        return row
    for p in points:
        controls.extend([control('gradient', p), control('direct_expm', p)])
    for p in product((0., 2*pi), (0., pi)):
        controls.append(control('direct_expm', p))
    for p in [(0., 0.), (.73, .33), (0., 0.)]:
        controls.extend([control('direct_expm', p), control('candidate_reevaluation', p)])
    controls += [control('cyclic_transport', p) for p in product(np.linspace(0., 2*pi, 5), np.linspace(0., pi, 5))]
    reference = dict(candidates=candidates, attempts=attempts, refined=True, full_bundle=True,
                     coarse=value, pooled=value, route_a=value, route_b=value, r=0., f=0.)
    coarse_reference=dict(candidates=[c for c in candidates if c['source'] in ('grid','de_0')])
    return dict(bindings=bound, controls=controls, reference=reference, faces=faces, mixer_width=8.,
                coarse_reference=coarse_reference, interior_points=points, unresolved_reason=None, integrity_failure=False)


def unknown_rows():
    specs={(s['family'],s['replicate']):s for s in config.panel()}
    return [dict(family=f, replicate=r, spec=specs[f,r], Q=None, c=None, arms={a: dict(bindings=None, reference=None,
        controls=[], unresolved_reason='unexecuted', integrity_failure=False) for a in ('low', 'high')})
        for f in range(4) for r in range(8)]


class SelectionTests(unittest.TestCase):
    def test_exact_cycle_enumeration_and_automorphisms(self):
        cycles = selection.cycles()
        self.assertEqual(len(selection.PERMUTATIONS), 720)
        self.assertEqual(len(cycles), 60)
        counts = Counter(selection.relabel(selection.BASE_EDGES, p) for p in selection.PERMUTATIONS)
        self.assertEqual(set(counts.values()), {12})
        for edges in cycles:
            self.assertEqual(Counter(v for e in edges for v in e), Counter({i: 2 for i in range(6)}))
            MixerSpec('cycle', 6, edges, 'synthetic')  # constructor verifies connectivity

    def test_exact_transport_and_uniform_invariance_all_sixty(self):
        basis = toy().feasible_basis()
        index = {tuple(x): i for i, x in enumerate(basis)}
        h = xy_mixer_hamiltonian(basis, MixerSpec('base', 6, selection.BASE_EDGES, 'synthetic'))
        spectrum = np.linalg.eigvalsh(h)
        uniform = np.full(20, 1/np.sqrt(20))
        for edges, p in selection.cycles().items():
            inverse = np.argsort(p)
            mapping = [index[tuple(x[list(p)])] for x in basis]
            other = xy_mixer_hamiltonian(basis, MixerSpec('relabel', 6, edges, 'synthetic'))
            np.testing.assert_array_equal(other, h[np.ix_(mapping, mapping)])
            np.testing.assert_array_equal(uniform[mapping], uniform)
            np.testing.assert_allclose(np.linalg.eigvalsh(other), spectrum, atol=1e-13, rtol=0)

    def test_selection_is_coefficient_only_reproducible(self):
        with patch('struct2circuit.simulator.FeasibleSubspaceQAOA.state_dimensionless', side_effect=AssertionError('outcome leakage')):
            first, second = selection.select(toy()), selection.select(toy())
        self.assertEqual(first, second)
        self.assertEqual(first['status'], 'selected')
        self.assertGreater(first['selected']['high']['selection_score'], first['selected']['low']['selection_score'])
        for arm, fn in [('low', min), ('high', max)]:
            eligible = [r for r in first['enumeration'] if r['edges'] == r['representative']]
            target = fn(r['selection_score'] for r in eligible)
            self.assertEqual(first['selected'][arm]['edges'], min(r['edges'] for r in eligible if r['selection_score'] == target))

    def test_exact_cost_symmetry_aliases_and_constant(self):
        row = selection.select(toy(True))
        self.assertEqual(len(row['cost_symmetries']), 720)
        self.assertEqual(len({r['representative'] for r in row['enumeration']}), 1)
        self.assertEqual(row['status'], 'NO_ALIGNMENT_CONTRAST')
        self.assertFalse(row['selected'])

    def test_partial_symmetry_and_nonconstant_score_tie(self):
        p = CardinalityQUBO(np.zeros((6, 6)), [.25]*3+[.75]*3, 3)
        row = selection.select(p)
        self.assertEqual(len(row['cost_symmetries']), 36)
        self.assertEqual(len(row['enumeration']), 60)
        self.assertTrue(all(r['representative'] <= r['edges'] for r in row['enumeration']))

    def test_rms_against_independent_exact_swap_enumeration(self):
        p = toy()
        row = selection.select(p)
        occupied, costs, _, _ = selection.exact_costs(p)
        lookup = dict(zip(occupied, costs))
        rms = {}
        for i, j in combinations(range(6), 2):
            changes = [lookup[tuple(sorted(set(s)-{i}|{j}))]-lookup[s] for s in occupied if i in s and j not in s]
            rms[i, j] = np.sqrt(float(sum(v*v for v in changes)/len(changes)))
        for graph in row['enumeration']:
            expected = sum(rms[e] for e in graph['edges'])/(6*row['span'])
            self.assertAlmostEqual(graph['score'], expected, places=14)

    def test_tiny_units_do_not_make_nonconstant_constant(self):
        p = toy()
        tiny = selection.select(CardinalityQUBO(p.Q*2**-40, p.c*2**-40, 3))
        original = selection.select(p)
        self.assertEqual(tiny['status'], 'selected')
        self.assertEqual(tiny['selected']['high']['edges'], original['selected']['high']['edges'])

    def test_rounded_score_tie_retains_unknown_without_replacement(self):
        with patch('struct2circuit.alignment_experiment.selection.round',return_value=0.):
            row=selection.select(toy())
        self.assertEqual(row['status'],'NO_ALIGNMENT_CONTRAST')
        self.assertEqual(len(row['enumeration']),60)
        self.assertFalse(row['selected'])

    def test_identity_binds_qubo_graph_and_uniform_state(self):
        p = toy(); s = selection.select(p)
        a = evidence.bindings(p, s['selected']['low']['edges'], 'toy', 'implementation')
        b = evidence.bindings(p, s['selected']['high']['edges'], 'toy', 'implementation')
        self.assertNotEqual(a['objective_identity'], b['objective_identity'])
        self.assertEqual(a['state_sha256'], b['state_sha256'])
        changed = CardinalityQUBO(p.Q, p.c+np.arange(6)/16, 3)
        self.assertNotEqual(evidence.bindings(changed, s['selected']['low']['edges'], 'toy', 'implementation')['objective_identity'], a['objective_identity'])


class MetadataTests(unittest.TestCase):
    def test_panel_and_seed_domains(self):
        rows = config.panel()
        self.assertEqual(len(rows), 32)
        self.assertEqual(Counter(r['family'] for r in rows), {f: 8 for f in range(4)})
        self.assertEqual(len({v for r in rows for v in r['seeds'].values()}), 160)
        self.assertEqual(rows, config.panel())
        self.assertTrue(all(r['n'] == 6 and r['k'] == 3 for r in rows))
        with self.assertRaises(ValueError): config.seed('validation', 0, 0)

    def test_metadata_entry_point_never_generates(self):
        entry = config.ROOT/'experiments/run_controlled_alignment.py'
        result = subprocess.run([sys.executable, str(entry)], capture_output=True, text=True, check=True)
        value = json.loads(result.stdout)
        self.assertEqual(value['manifest']['status'], 'NOT_EXECUTED')
        self.assertEqual(len(value['manifest']['panel']), 32)

    def test_approved_design_and_old_protocols(self):
        config.verify_design()
        self.assertEqual(hashlib.sha256(config.DESIGN.read_bytes()).hexdigest(), config.DESIGN_HASH)

    def test_wrong_thread_environment_and_runtime_rejected(self):
        with patch.dict(os.environ,{'OPENBLAS_NUM_THREADS':'2'}):
            with self.assertRaises(ValueError): config.runtime()
        wrong=config.identity() | dict(numpy='wrong-version')
        with patch.object(config,'identity',return_value=wrong):
            with self.assertRaises(ValueError): config.runtime()

    def test_new_panel_seed_collision_checks_public_allowlist(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'configs').mkdir();(root/'results').mkdir()
            (root/'configs/benchmark_v1_draft.json').write_text(json.dumps(dict(master_seed=3,splits={},random_baseline_replicates=1)))
            for name in ('pilot_results.csv','independent_dks_benchmark_v1_instances.csv'):
                (root/'results'/name).write_text('problem_seed\n123\n')
            with patch.object(config,'panel',return_value=[dict(seeds=dict(qubo=123))]):
                with self.assertRaises(ValueError): config.public_seed_check(root)

    def test_clean_inherited_source_hash(self):
        from struct2circuit.calibration.storage import driver_identity
        self.assertEqual(driver_identity()['sha256'],'c1d0419129380837feac5f100617f3e0c00bbbb0cb9f3fb1f37575f472b9cfb2')


class DecisionTests(unittest.TestCase):
    def test_exact_material_boundaries(self):
        self.assertEqual(evidence.classify(-.00012), 'neutral')
        self.assertEqual(evidence.classify(.00012), 'neutral')
        self.assertEqual(evidence.classify(np.nextafter(-.00012, -np.inf)), 'win')
        self.assertEqual(evidence.classify(np.nextafter(.00012, np.inf)), 'loss')
        self.assertEqual(evidence.classify(None), 'unknown')

    def test_all_561_primary_count_boundaries(self):
        for w in range(33):
            for u in range(33-w):
                result = evidence.primary(dict(win=w, unknown=u, loss=32-w-u, neutral=0))
                self.assertEqual(result['outcome'], 'DIRECTIONAL_SUPPORT' if w >= 24 else 'CONTRADICTION' if w+u <= 8 else 'INCONCLUSIVE')
                self.assertEqual(result['lower'] > .5, w >= 24)
                self.assertEqual(result['upper'] < .5, w+u <= 8)

    def test_denominator_and_family_missingness(self):
        rows = unknown_rows()
        result = evidence.analyze(rows)
        self.assertEqual(result['counts']['unknown'], 32)
        self.assertEqual(result['outcome'], 'INCONCLUSIVE')
        self.assertTrue(all(r['unknown'] == 8 for r in result['families'].values()))
        for bad in [rows[:-1], rows+[rows[0]], [rows[0]]*32]:
            with self.assertRaises(ValueError): evidence.analyze(bad)
        with self.assertRaises(ValueError): evidence.primary(dict(win=64, loss=0, neutral=0, unknown=0))

    def test_integrity_failure_overrides_favorable_results(self):
        value = evidence.primary(dict(win=32, loss=0, neutral=0, unknown=0), integrity_failure=True)
        self.assertEqual(value['outcome'], 'INCONCLUSIVE')


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid = scalar_arm()

    def test_complete_synthetic_scalar_reference(self):
        value, failure = evidence.verify_reference(self.valid, 0, 0)
        self.assertEqual(value, .2)
        self.assertFalse(failure)

    def test_flags_do_not_replace_refinement_attempts(self):
        for source in ('fine_lbfgsb', 'powell', 'de_1', 'de_polish_1'):
            row = deepcopy(self.valid)
            row['reference']['attempts'] = [a for a in row['reference']['attempts'] if a['source'] != source]
            self.assertIsNone(evidence.verify_reference(row, 0, 0)[0])

    def test_wrong_start_and_powell_duplicate_are_unresolved(self):
        for source in ('fine_lbfgsb', 'powell'):
            row = deepcopy(self.valid)
            next(a for a in row['reference']['attempts'] if a['source'] == source)['start'] = [.123, .234]
            self.assertIsNone(evidence.verify_reference(row, 0, 0)[0])

    def test_missing_grid_or_face_work_is_unresolved(self):
        row = deepcopy(self.valid); row['reference']['candidates'].pop(0)
        self.assertIsNone(evidence.verify_reference(row, 0, 0)[0])
        row = deepcopy(self.valid); row['faces'][0]['complete'] = False
        self.assertIsNone(evidence.verify_reference(row, 0, 0)[0])

    def test_scalar_summary_and_route_seed_substitution_rejected(self):
        row = deepcopy(self.valid); row['reference']['pooled'] = .1
        with self.assertRaises(ValueError): evidence.verify_reference(row, 0, 0)
        row = deepcopy(self.valid); next(a for a in row['reference']['attempts'] if a['source'] == 'de_0')['seed'] = 1
        with self.assertRaises(ValueError): evidence.verify_reference(row, 0, 0)

    def test_control_and_candidate_identity_substitution_rejected(self):
        for target in ('controls', 'candidates'):
            row = deepcopy(self.valid)
            if target == 'controls': row['controls'][0]['bindings'] = dict(objective_identity='another-landscape')
            else: row['reference']['candidates'][0]['objective_identity'] = 'another-landscape'
            with self.assertRaises(ValueError): evidence.verify_reference(row, 0, 0)

    def test_control_coverage_is_required(self):
        for kind in ('gradient', 'direct_expm', 'cyclic_transport', 'candidate_reevaluation'):
            row = deepcopy(self.valid); row['controls'] = [c for c in row['controls'] if c['kind'] != kind]
            self.assertIsNone(evidence.verify_reference(row, 0, 0)[0])

    def test_negative_control_failures_survive_following_landscape(self):
        row = deepcopy(self.valid)
        row['controls'][0].update(error=2e-6, analytic=[2e-6, 0.], status='fail')
        self.assertEqual(evidence.verify_reference(row, 0, 0), (None, True))
        self.assertEqual(evidence.verify_reference(self.valid, 0, 0), (.2, False))

    def test_unresolved_and_budget_interrupted_work_cannot_pass(self):
        row = deepcopy(self.valid); row['unresolved_reason'] = 'arm_evaluation_ceiling'
        self.assertIsNone(evidence.verify_reference(row, 0, 0)[0])
        row = deepcopy(self.valid); row['reference']['attempts'][0]['interrupted'] = 'ArmExhausted'
        self.assertIsNone(evidence.verify_reference(row, 0, 0)[0])


class ResourceTests(unittest.TestCase):
    def test_native_cache_hits_and_fresh_calls_with_nonquantum_double(self):
        p=toy();basis=p.feasible_basis();state=np.full(20,1/np.sqrt(20),dtype=complex)
        calls=[]
        sim=SimpleNamespace(problem=p,basis=basis,mixer_hamiltonian=np.eye(20),initial_state=state,
            normalized_costs=np.linspace(0.,1.,20),feasible_span=2.,_mixer_eigenvalues=np.arange(20),
            state_dimensionless=lambda *args: calls.append(args) or state.copy(),
            _evaluate_state=lambda _:SimpleNamespace(normalized_gap=.2,expectation=.4,cost_status='nonconstant'))
        with tempfile.TemporaryDirectory() as d:
            b=Budget(Path(d)/'journal',cpu_clock=lambda:0.)
            obj=Objective(sim,b,'toy')
            obj.evaluate(.2,.3);obj.evaluate(.2,.3);obj.evaluate(.2,.3,fresh=True)
            self.assertEqual(b.counts['actual_evaluations'],2)
            self.assertEqual(b.counts['cache_hits'],1)
            self.assertEqual(len(calls),2);b.close()
    def test_actual_cache_and_derivative_accounting_without_quantum_simulation(self):
        with tempfile.TemporaryDirectory() as d:
            b = Budget(Path(d)/'journal', total=100, arm=100, cpu_clock=lambda: 0.)
            obj = object.__new__(Objective)
            obj.identity, obj.budget, obj.landscape = 'toy', b, 'arm'
            with patch.object(obj, 'evaluate', return_value=None), patch.object(NativeObjective, 'gradient', return_value=np.zeros(2)):
                obj.gradient([.2, .3], 'B00'); obj.gradient([.2, .3], 'B00')
            self.assertEqual(b.counts['actual_evaluations'], 2)
            self.assertEqual(b.counts['derivative_evolutions'], 2)
            b.close()

    def test_total_ceiling_cannot_be_hidden_by_new_arm(self):
        with tempfile.TemporaryDirectory() as d:
            b = Budget(Path(d)/'journal', total=2, arm=2, cpu_clock=lambda: 0.)
            for name in ('first', 'second'): b.charge((name, pack('>dd', 0., 0.)), name, 'B00')
            with self.assertRaises(BudgetExhausted): b.charge(('third', pack('>dd', 0., 0.)), 'third', 'B00')
            self.assertEqual(b.stop_reason, 'total_evaluation_ceiling'); b.close()

    def test_exact_20000_arm_ceiling_and_other_arm_capacity(self):
        with tempfile.TemporaryDirectory() as d:
            b = Budget(Path(d)/'journal', cpu_clock=lambda: 0.)
            for i in range(20000): b.charge(('toy', pack('>dd', float(i), 0.)), 'arm', 'B00')
            with self.assertRaises(ArmExhausted): b.charge(('toy', pack('>dd', 0., 0.)), 'arm', 'B00')
            b.charge(('other', pack('>dd', 0., 0.)), 'another-arm', 'B00')
            self.assertEqual(b.counts['actual_evaluations'], 20001)
            self.assertEqual(b.arm_stops['arm'], 'arm_evaluation_ceiling')
            b.close()

    def test_cpu_cutoff_and_reporting_reserve(self):
        with tempfile.TemporaryDirectory() as d:
            clock = [3299.]
            b = Budget(Path(d)/'journal', cpu_clock=lambda: clock[0])
            b.check_time(); clock[0] = 3300.
            with self.assertRaises(BudgetExhausted): b.check_time()
            self.assertEqual(b.snapshot()['limits']['cpu'], 3600.)
            b.close()

    def test_restarts_are_rejected_in_new_process(self):
        with tempfile.TemporaryDirectory() as d:
            journal = Path(d)/'journal'
            b = Budget(journal, cpu_clock=lambda: 0.); b.charge(('toy', pack('>dd', 0., 0.)), 'arm', 'B00'); b.close()
            code = 'from struct2circuit.alignment_experiment.budget import Budget; Budget(__import__("sys").argv[1])'
            env = dict(os.environ, PYTHONPATH=str(config.ROOT/'src'))
            result = subprocess.run([sys.executable, '-c', code, str(journal)], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('FileExistsError', result.stderr)
            self.assertEqual(len(journal.read_text().splitlines()), 1)

    def test_journal_and_wave_receipts_reject_tampering_or_counter_reset(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); b = Budget(root/'evaluations.jsonl', cpu_clock=lambda: 0.)
            b.charge(('toy', pack('>dd', 0., 0.)), 'arm', 'B00')
            first = storage.checkpoint(root, b, 'first')
            b.charge(('toy', pack('>dd', 1., 0.)), 'arm', 'B00')
            second = storage.checkpoint(root, b, 'second', previous=first); b.close()
            self.assertEqual(storage.verify_ledger(root, second)['counters']['actual_evaluations'], 2)
            path = root/second; record = json.loads(path.read_text())
            record['payload']['snapshot']['per_arm']['arm'] = 0
            record['sha256'] = canonical_hash(record['payload']); path.write_text(json.dumps(record))
            with self.assertRaises(ValueError): storage.verify_ledger(root, second)

    def test_resource_guard_memory_disk_and_unknown_information(self):
        with self.assertRaises(BudgetExhausted): Guard(Path('/tmp'), sample=lambda _: (8*1024**3, 6*1024**3, 4*1024**3, 10*1024**3))(launch=True)
        with self.assertRaises(BudgetExhausted): Guard(Path('/tmp'), sample=lambda _: (8*1024**3, 6*1024**3, 1, 4*1024**3))(launch=True)
        with self.assertRaises(BudgetExhausted): Guard(Path('/tmp'), sample=lambda _: (_ for _ in ()).throw(OSError()))(launch=True)

    def test_counter_domain_cpu_and_limit_resets_fail_even_with_new_digest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); clock=[0.]
            b=Budget(root/'evaluations.jsonl',cpu_clock=lambda:clock[0],wall_clock=lambda:clock[0])
            clock[0]=1.;b.charge(('toy',pack('>dd',0.,0.)),'arm','B00',derivative=True)
            first=storage.checkpoint(root,b,'first')
            clock[0]=2.;b.charge(('toy',pack('>dd',1.,0.)),'arm','B00',oracle=True)
            last=storage.checkpoint(root,b,'last',previous=first);b.close()
            original=json.loads((root/last).read_text())
            changes=[lambda s:s['counters'].update(actual_evaluations=1),
                lambda s:s['counters'].update(oracle_calls=0),
                lambda s:s['counters'].update(derivative_evolutions=0),
                lambda s:s['per_arm'].update(arm=0),
                lambda s:s['per_domain'][0].update(count=0),
                lambda s:s.update(cpu_seconds=.5),
                lambda s:s.update(wall_seconds=.5),
                lambda s:s['limits'].update(arm=20001)]
            for change in changes:
                record=deepcopy(original);change(record['payload']['snapshot'])
                record['sha256']=canonical_hash(record['payload'])
                (root/last).write_text(json.dumps(record))
                with self.assertRaises(ValueError):storage.verify_ledger(root,last)


class SafetyTests(unittest.TestCase):
    def test_absent_mismatched_and_reused_authorization(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaisesRegex(PermissionError, 'Separate persisted'): storage.authorization(None, root/'output')
            auth = root/'auth.json'; auth.write_text('{}')
            with self.assertRaisesRegex(PermissionError, 'Authorization must bind'): storage.authorization(auth, root/'output')
            value = dict(execution_authorized=True, scope=config.SCOPE, design_sha256=config.DESIGN_HASH,
                implementation=config.identity(), manifest_sha256=canonical_hash(config.manifest()), output=str(root/'output'),
                approval_identity='synthetic-test-only', historical_archive='/workspace/calibration-archives/stage7b-development-e8dea4d-v1.tar.gz',
                historical_archive_sha256=config.HISTORY_HASH,
                development_records={str(Path('/workspace/calibration-runs')/n/'instances.jsonl'): h for n, h in config.DEVELOPMENT.items()})
            auth.write_text(json.dumps(value)); Path(str(auth)+'.used').write_text('already-used')
            historical = value['development_records']
            historical_text = json.dumps(dict(instance_id=0, split='development', coefficients_sha256='synthetic-toy'))
            read_text = Path.read_text
            def fixture_hash(p):
                if str(p) in historical: return historical[str(p)]
                if str(p).endswith('.tar.gz'): return config.HISTORY_HASH
                return hashlib.sha256(Path(p).read_bytes()).hexdigest()
            def fixture_text(p, *args, **kwargs):
                if str(p) in historical: return historical_text
                return read_text(p, *args, **kwargs)
            with patch.object(storage, 'file_hash', side_effect=fixture_hash), patch.object(Path, 'read_text', new=fixture_text), patch.object(storage, 'public_seed_check'):
                with self.assertRaisesRegex(PermissionError, 'already consumed'): storage.authorization(auth, root/'output')

    def test_authorization_claim_is_one_use_and_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = root/'auth'; p.write_text('{"approval_identity": "synthetic"}\n')
            auth = json.loads(p.read_text()); storage.claim(p, root/'out', auth)
            self.assertEqual((root/'out/authorization.json').read_bytes(), p.read_bytes())
            with self.assertRaises(FileExistsError): storage.claim(p, root/'another', auth)
            self.assertFalse((root/'another').exists())

    def test_fresh_path_preflight_protects_sentinels_and_aliases(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); sentinel = root/'existing'; sentinel.write_bytes(b'unchanged')
            for p in [sentinel, root/'missing/../existing']:
                with self.assertRaises(FileExistsError): storage.fresh_output(p)
            link = root/'link'; link.symlink_to(root/'absent')
            with self.assertRaises(FileExistsError): storage.fresh_output(link)
            parent = root/'alias'; parent.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError): storage.fresh_output(parent/'fresh')
            self.assertEqual(sentinel.read_bytes(), b'unchanged')
            self.assertEqual(storage.fresh_output(root/'fresh'), root/'fresh')
        with self.assertRaises(ValueError): storage.fresh_output(config.ROOT/'results/new-study')
        with self.assertRaises(ValueError): storage.fresh_output(Path('/workspace/calibration-runs/new-study'))

    def test_invalid_entrypoint_rejects_without_output(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)/'output'
            result = subprocess.run([sys.executable, str(config.ROOT/'experiments/run_controlled_alignment.py'), '--execute', '--output', str(target)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(target.exists())


class WorkflowTests(unittest.TestCase):
    def scalar_objective(self, budget):
        """A constant scalar oracle double; it never calls a quantum simulator."""
        h = xy_mixer_hamiltonian(toy().feasible_basis(), MixerSpec('cycle', 6, selection.BASE_EDGES, 'synthetic'))
        class Scalar:
            identity, landscape = 'toy-identity', 'toy-arm'
            simulator = SimpleNamespace(cost_status='nonconstant', initial_state=np.full(20,1/np.sqrt(20)), mixer_hamiltonian=h)
            def __init__(self): self.budget, self.cache = budget, {}
            def key(self, u, beta): return self.identity, pack('>dd', float(u), float(beta))
            def evaluate(self, u, beta, domain='B00', *, fresh=False):
                key = self.key(u, beta); budget.request(self.landscape, domain)
                if not fresh and key in self.cache:
                    budget.counts['cache_hits'] += 1; return self.cache[key]
                budget.charge(key, self.landscape, domain)
                self.cache[key] = SimpleNamespace(gap=.2)
                return self.cache[key]
            def gradient(self, point, domain):
                budget.charge(self.key(*point), self.landscape, domain, derivative=True)
                return np.zeros(2)
        return Scalar()

    def test_full_schedule_with_scalar_double_and_stub_solvers(self):
        observed_de = []
        def local(fun, start, *, method, bounds, options, jac=None):
            fun(start)
            if jac: jac(start)
            return SimpleNamespace(x=np.array(start), success=True, message='synthetic solver stub')
        def de(fun, bounds, **kw):
            observed_de.append(kw)
            fun([.73, .33])
            return SimpleNamespace(x=np.array([.73,.33]), success=True, message='synthetic DE stub')
        with tempfile.TemporaryDirectory() as d:
            b = Budget(Path(d)/'journal', cpu_clock=lambda: 0.)
            obj = self.scalar_objective(b); search = ReferenceSearch(obj, 0, 0)
            with patch('struct2circuit.calibration.reference.minimize', side_effect=local), patch('struct2circuit.alignment_experiment.reference.minimize', side_effect=local), patch('struct2circuit.alignment_experiment.reference.differential_evolution', side_effect=de):
                result = search.fine(search.coarse())
            row = scalar_arm(); row['reference'] = json.loads(json.dumps(result.record()))
            row['coarse_reference'] = dict(candidates=[c for c in row['reference']['candidates'] if c['source'] in ('grid','lbfgsb','de_0','de_polish_0')])
            row['faces'] = json.loads(json.dumps(search.face_records))
            self.assertEqual(evidence.verify_reference(row,0,0), (.2,False))
            self.assertEqual(len(observed_de), 2)
            self.assertEqual([x['maxiter'] for x in observed_de], [63,126])
            self.assertTrue(all(isinstance(x['seed'], np.random.Generator) for x in observed_de))
            self.assertEqual(Counter(a['source'] for a in result.attempts)['fine_lbfgsb'], 8)
            self.assertEqual(len(search.face_records), 8)
            self.assertLess(b.counts['actual_evaluations'], 9000)
            b.close()

    def test_grid_cap_interrupts_reference_without_falsely_completing(self):
        with tempfile.TemporaryDirectory() as d:
            b = Budget(Path(d)/'journal', cpu_clock=lambda: 0.)
            for _ in range(19999): b.charge(('toy-identity',pack('>dd',0.,0.)),'toy-arm','B00')
            search = ReferenceSearch(self.scalar_objective(b), 0, 0)
            result = search.coarse()
            self.assertFalse(result.complete)
            self.assertIn('ceiling', result.unresolved_reason)
            self.assertFalse(search.fine(result).complete)
            self.assertEqual(b.counts['actual_evaluations'], 20000)
            b.close()

    def test_duplicate_identity_stops_before_simulator_construction(self):
        from struct2circuit.alignment_experiment import driver
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); auth=root/'auth'; auth.write_text('{"approval_identity":"synthetic-only"}')
            with patch.object(driver, 'runtime', return_value={}), patch.object(driver, 'public_seed_check', return_value={}), patch.object(driver, 'authorization', return_value=(json.loads(auth.read_text()),set())), patch.object(driver.Guard, '__call__'), patch.object(driver.problems,'block_correlated_qubo',return_value=toy()), patch.object(driver.problems,'weighted_densest_k_subgraph_qubo',return_value=toy()), patch.object(driver,'FeasibleSubspaceQAOA',side_effect=AssertionError('selection must precede simulation')):
                driver.run_once(auth,root/'out')
            result=json.loads((root/'out/provisional-analysis.json').read_text())
            self.assertEqual(result['outcome'],'INCONCLUSIVE')
            self.assertTrue(result['integrity_failure'])
            self.assertEqual(result['counts']['unknown'],32)
            self.assertEqual((root/'out/evaluations.jsonl').read_text(),'')

    def test_durable_inventory_verification_and_file_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); implementation=config.identity()
            auth=dict(execution_authorized=True,scope=config.SCOPE,design_sha256=config.DESIGN_HASH,
                implementation=implementation,manifest_sha256=canonical_hash(config.manifest()),output=str(root),
                approval_identity='synthetic-test-only', historical_archive='/workspace/calibration-archives/stage7b-development-e8dea4d-v1.tar.gz',
                historical_archive_sha256=config.HISTORY_HASH,
                development_records={str(Path('/workspace/calibration-runs')/n/'instances.jsonl'): h for n,h in config.DEVELOPMENT.items()})
            atomic_json(root/'authorization.json',auth)
            b=Budget(root/'evaluations.jsonl',cpu_clock=lambda: 0.)
            storage.finish(root,b,unknown_rows(),implementation,storage.file_hash(root/'authorization.json'),previous=None)
            b.close();atomic_json(root/'worker-ended.json',dict(cpu_seconds=.1,process_ended=True))
            storage.settle(root,.2)
            self.assertEqual(storage.verify(root)['analysis']['counts']['unknown'],32)
            self.assertFalse(storage.verify(root)['scientific_publication_authorized'])
            (root/'records.json').write_text('[]')
            with self.assertRaises(ValueError):storage.verify(root)

    def test_resource_sealing_cpu_is_included_in_settlement(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); atomic_json(root/'inventory.json',{})
            atomic_json(root/'worker-ended.json',dict(cpu_seconds=3599.5,process_ended=True))
            storage.settle(root,3600.1)
            value=json.loads(Path(str(root)+'.settled.json').read_text())
            self.assertEqual(value['stop_reason'],'cpu_ceiling')
            self.assertEqual(value['cpu_seconds'],3600.1)
            self.assertEqual(value['worker_receipt_sha256'],storage.file_hash(root/'worker-ended.json'))


if __name__ == '__main__':
    unittest.main()
