from __future__ import annotations

from dataclasses import asdict
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from struct2circuit.calibration.budget import Budget, BudgetExhausted, Limits
from struct2circuit.calibration.config import PROTOCOL_HASH, instance_specs
from struct2circuit.calibration.decisions import Check
from struct2circuit.calibration.driver import Case, PhaseDriver, generate_cases, require_execution_authorization, run_synthetic
from struct2circuit.calibration.resources import GIB, MIB, ResourceGuard, ResourceExhausted
from struct2circuit.calibration.storage import (ARTIFACTS, OutputStore, ValidationGate,
    canonical_json, create_validation_lock)
from struct2circuit.mixers import ring_mixer
from struct2circuit.problems import CardinalityQUBO
from struct2circuit.simulator import FeasibleSubspaceQAOA


@contextmanager
def healthy_resource_guard():
    """Keep entry-point safety tests independent of host resource discovery."""
    with patch('struct2circuit.calibration.driver.ResourceGuard',
               side_effect=lambda *args: ResourceGuard(*args, sample=lambda paths:
                   (32*GIB, 30*GIB, 64*MIB, 10*GIB))):
        yield


def finalized_mock_development(directory, budget, *, policy=17, reserves=False, status='ACCEPTED WITH EXPLICIT LIMITATIONS'):
    """Persist mock records only; no approved QUBO or numerical panel is generated."""
    from struct2circuit.calibration.config import build_manifest, SCOPE
    from struct2circuit.calibration.storage import driver_identity
    specs = [s for s in instance_specs(reserves) if s.split != 'validation']
    keys = [(s.instance_id, g, init) for s in specs for g in range(8) for init in range(3)]
    manifest = build_manifest(reserves) | dict(kind='SYNTHETIC_MOCK_FINALIZATION_TEST', phase='development',
        status='INCONCLUSIVE', phase_status=status, selected_policy=policy, development_finalized=True,
        validation_executed=False, expected_phase_keys=keys, provenance=dict(driver=driver_identity()))
    with OutputStore(directory) as store:
        store.json('calibration_manifest.json', manifest)
        for spec in specs: store.row('instances.jsonl', dict(instance_id=spec.instance_id, status='generated', mock=True))
        for key in keys:
            store.row('graphs.jsonl', dict(landscape=key, mock=True))
            store.row('landscapes.jsonl', dict(landscape=key, status='observed', error=None,
                arithmetic_failure=False, references={n:dict(complete=True) for n in ('B00','B10','B01','B11')}, mock=True))
        store.row('candidates.jsonl', dict(mock=True, official_calibration_executed=False))
        store.json('budget.json', budget.snapshot() | dict(incomplete_required_work=[]))
        store.decision('INCONCLUSIVE\n\n'+f'Phase numerical status: {status}.\nPolicy: P{policy} on B00 only.\n'
            + f'Scope: {SCOPE}. Validation completed: False. Synthetic check: False.')
        store.sync()
    return directory


class BudgetTests(unittest.TestCase):
    def make(self, **changes):
        return Budget(Limits(**(dict(essential=2, shared=2, total=4, landscape=4, optional=1, cpu_seconds=20)|changes)))

    def charge(self, budget, index, account='essential', optional=False):
        budget.request('toy', 'B00')
        budget.charge(('model', bytes([index])), 'toy', 'B00', account, optional=optional)

    def test_essential_spills_into_shared_without_increasing_total(self):
        b = self.make()
        for i in range(4): self.charge(b, i)
        self.assertEqual(b.counts['essential'], 2)
        self.assertEqual(b.counts['shared'], 2)
        self.assertEqual(b.counts['essential_to_shared'], 2)
        with self.assertRaises(BudgetExhausted): self.charge(b, 4)
        self.assertEqual(b.counts['actual_evaluations'], 4)

    def test_landscape_ceiling_prevents_overrun(self):
        b = self.make(landscape=1)
        self.charge(b, 0)
        with self.assertRaises(BudgetExhausted): self.charge(b, 1)
        self.assertEqual(b.landscapes['toy'], 1)
        self.assertEqual(b.stop_reason, 'landscape_evaluation_ceiling')

    def test_optional_is_subordinate_and_capped(self):
        b = self.make()
        with self.assertRaises(ValueError): self.charge(b, 0, 'shared', True)
        self.assertEqual(b.counts['actual_evaluations'], 0)
        b.mandatory_complete = True
        self.charge(b, 0, 'shared', True)
        with self.assertRaises(BudgetExhausted): self.charge(b, 1, 'shared', True)
        self.assertEqual(b.counts['optional'], 1)
        self.assertEqual(b.counts['shared'], 1)
        self.assertEqual(b.stop_reason, 'optional_evaluation_ceiling')

    def test_cpu_stop_is_sticky(self):
        clock = [0.]
        b = Budget(Limits(cpu_seconds=1), cpu_clock=lambda: clock[0], wall_clock=lambda: clock[0])
        clock[0] = 1.
        with self.assertRaises(BudgetExhausted): b.request('toy', 'B00')
        clock[0] = 0.
        with self.assertRaises(BudgetExhausted): b.request('toy', 'B00')
        self.assertEqual(b.counts['actual_evaluations'], 0)

    def test_persisted_budget_does_not_reset_on_validation(self):
        b = self.make()
        self.charge(b, 0)
        self.charge(b, 1)
        snapshot = json.loads(canonical_json(b.snapshot()))
        restored = Budget.from_snapshot(snapshot)
        self.assertEqual(restored.counts['actual_evaluations'], 2)
        self.assertEqual(len(restored.unique), 2)
        self.charge(restored, 2)
        self.assertEqual(restored.counts['shared'], 1)
        snapshot['counters']['actual_evaluations'] = 100
        with self.assertRaises(ValueError): Budget.from_snapshot(snapshot)

    def test_approved_limits_cannot_be_increased(self):
        with self.assertRaises(ValueError): Limits(total=8_000_001)
        with self.assertRaises(ValueError): Limits(cpu_seconds=28_801)
        with self.assertRaises(ValueError): Limits(total=0)
        with self.assertRaises(ValueError): Limits(total=True)

    def test_corrupt_persisted_cpu_or_counts_cannot_bypass_limits(self):
        for field, value in (('cpu_seconds', float('nan')), ('wall_seconds', -1)):
            snapshot = self.make().snapshot()
            snapshot[field] = value
            with self.assertRaises(ValueError): Budget.from_snapshot(snapshot)
        snapshot = self.make().snapshot()
        snapshot['counters']['actual_evaluations'] = 0.5
        with self.assertRaises(ValueError): Budget.from_snapshot(snapshot)


class OutputTests(unittest.TestCase):
    def test_fresh_artifacts_and_null_serialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)/'fresh'
            with OutputStore(out) as store:
                store.json('calibration_manifest.json', dict(kind='SYNTHETIC', metric=None))
                store.row('landscapes.jsonl', dict(status='missing', gap=float('nan')))
                store.json('budget.json', dict(status='interrupted'))
                store.decision('INCONCLUSIVE')
            self.assertEqual({p.name for p in out.iterdir()}, set(ARTIFACTS))
            row = json.loads((out/'landscapes.jsonl').read_text())
            self.assertIsNone(row['gap'])

    def test_existing_artifact_is_preserved_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root/'occupied'
            out.mkdir()
            sentinel = out/'calibration_manifest.json'
            sentinel.write_bytes(b'original\x00evidence')
            before = sentinel.read_bytes()
            with self.assertRaises(FileExistsError): OutputStore(out)
            self.assertEqual(sentinel.read_bytes(), before)
            self.assertEqual(list(out.iterdir()), [sentinel])

    def test_dangling_terminal_link_and_symlink_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dangling = root/'dangling'
            dangling.symlink_to(root/'absent')
            with self.assertRaises(FileExistsError): OutputStore(dangling)
            real = root/'real'
            real.mkdir()
            alias = root/'alias'
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaises(FileExistsError): OutputStore(alias/'new')
            self.assertFalse((real/'new').exists())

    def test_traversal_and_historical_namespace_before_any_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            historical = root/'historical'
            historical.mkdir()
            sentinel = historical/'decision.md'
            sentinel.write_bytes(b'unchanged')
            with self.assertRaises(FileExistsError): OutputStore(root/'missing'/'..'/'historical')
            self.assertFalse((root/'missing').exists())
            with self.assertRaises(FileExistsError): OutputStore(historical/'new', historical_root=historical)
            self.assertEqual(sentinel.read_bytes(), b'unchanged')
            self.assertFalse((historical/'new').exists())

    def test_hardlink_collision_and_partially_occupied_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sentinel = root/'sentinel'
            sentinel.write_bytes(b'original')
            out = root/'out'
            out.mkdir()
            (out/'budget.json').hardlink_to(sentinel)
            with self.assertRaises(FileExistsError): OutputStore(out)
            self.assertEqual(sentinel.read_bytes(), b'original')
            other = root/'partial'
            other.mkdir()
            (other/'unrelated').write_bytes(b'x')
            with self.assertRaises(FileExistsError): OutputStore(other)


class LockTests(unittest.TestCase):
    def create(self, path, *, policy=17, reserves=False, budget=None):
        budget = budget or Budget()
        budget.mandatory_complete = True
        evidence = finalized_mock_development(path.with_name(path.stem+'-development'), budget, policy=policy, reserves=reserves)
        return create_validation_lock(path, policy=policy, scope='bounded_B00_native_common_physical_beta',
            reserves=reserves, budget=budget, development_status='ACCEPTED WITH EXPLICIT LIMITATIONS', development_output=evidence)

    def test_missing_lock_and_generation_without_authorization(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError): ValidationGate(Path(temporary)/'missing.json')
        with self.assertRaises(PermissionError): require_execution_authorization(None, 'development')
        with patch('struct2circuit.problems.block_correlated_qubo', side_effect=AssertionError('No study QUBOs')):
            with self.assertRaises(PermissionError): generate_cases(instance_specs(), Budget(), authorization=None, phase='development')

    def test_lock_rejects_unresolved_development_and_modified_policy_without_reserves(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'lock.json'
            with self.assertRaises(ValueError): create_validation_lock(path, policy=17, scope='bounded', reserves=False,
                budget=Budget(), development_status='INCONCLUSIVE')
            self.assertFalse(path.exists())
            with self.assertRaises(ValueError): self.create(path, policy=9)
            self.assertFalse(path.exists())

    def test_immutable_lock_policy_and_one_validation_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'lock.json'
            gate = self.create(path)
            gate.authorize(17)
            gate.authorize(17)
            with self.assertRaises(ValueError): gate.authorize(9)
            another = ValidationGate(path)
            with self.assertRaises(FileExistsError): another.authorize(17)
            path.write_text(path.read_text()+' ')
            with self.assertRaises(ValueError): gate.authorize(17)

    def test_checksum_and_configuration_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'lock.json'
            self.create(path)
            document = json.loads(path.read_text())
            document['payload']['thresholds']['deficit'] = .1
            path.write_text(canonical_json(document))
            with self.assertRaises(ValueError): ValidationGate(path)
            document['sha256'] = sha256(canonical_json(document['payload']).encode()).hexdigest()
            path.write_text(canonical_json(document))
            with self.assertRaises(ValueError): ValidationGate(path)

    def test_validation_barrier_before_generator_even_with_execution_authorization(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'authorization.json'
            path.write_text(json.dumps(dict(calibration_execution_authorized=True, protocol_sha256=PROTOCOL_HASH, phases=['validation'])))
            specs = [s for s in instance_specs() if s.split == 'validation']
            with patch('struct2circuit.problems.block_correlated_qubo', side_effect=AssertionError('Holdout must not be generated')):
                with self.assertRaises(PermissionError): generate_cases(specs, Budget(), authorization=path, phase='validation')

    def test_in_memory_lock_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            gate = self.create(Path(temporary)/'lock.json')
            gate.payload['policy'] = 9
            with self.assertRaisesRegex(ValueError, 'altered'): gate.authorize(9)
            self.assertFalse((Path(temporary)/'lock.json.validation-started').exists())

    def test_holdout_generation_cannot_repeat_using_same_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            gate = self.create(Path(temporary)/'lock.json')
            gate.begin_generation(17)
            with self.assertRaisesRegex(ValueError, 'once'): gate.begin_generation(17)

    def test_software_version_change_invalidates_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'lock.json'
            self.create(path)
            document = json.loads(path.read_text())
            document['payload']['driver']['software']['scipy'] = 'different-version'
            document['sha256'] = sha256(canonical_json(document['payload']).encode()).hexdigest()
            path.write_text(canonical_json(document))
            with self.assertRaises(ValueError): ValidationGate(path)


class WorkflowTests(unittest.TestCase):
    def fixture(self, constant=False, sentinel=False):
        problem = CardinalityQUBO(np.zeros((4, 4)), np.ones(4) if constant else np.array([0., 0., 1., 1.]), 2)
        sim = FeasibleSubspaceQAOA(problem, ring_mixer(4))
        return Case(1042, 0, 0, 0, 0, sim, 64, sentinel=sentinel)

    def test_tiny_interrupted_workflow_and_missing_record(self):
        case = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)/'synthetic'
            limits = Limits(essential=8, shared=2, total=10, landscape=10, optional=1, cpu_seconds=10)
            with healthy_resource_guard():
                status, snapshot = run_synthetic([case], out, limits=limits, expected_keys=[case.key, (1043, 0, 0)])
            self.assertEqual(status, 'INCONCLUSIVE')
            self.assertEqual(snapshot['counters']['actual_evaluations'], 10)
            rows = [json.loads(row) for row in (out/'landscapes.jsonl').read_text().splitlines()]
            self.assertTrue(any(r['status'] == 'missing' for r in rows))
            self.assertEqual(json.loads((out/'calibration_manifest.json').read_text())['kind'], 'SYNTHETIC_IMPLEMENTATION_CHECK')

    def test_constant_workflow_keeps_null_and_never_runs_solver(self):
        with tempfile.TemporaryDirectory() as temporary:
            with healthy_resource_guard(), patch('struct2circuit.calibration.reference.minimize', side_effect=AssertionError('No constant solver')):
                status, snapshot = run_synthetic([self.fixture(True)], Path(temporary)/'synthetic', limits=Limits())
            self.assertEqual(status, 'INCONCLUSIVE')
            self.assertEqual(snapshot['counters'].get('actual_evaluations', 0), 0)

    def test_full_toy_workflow_all_primary_boxes_and_controls(self):
        case = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)/'synthetic'
            limits = Limits(essential=60000, shared=10000, total=70000, landscape=70000, optional=10, cpu_seconds=30)
            with healthy_resource_guard():
                status, snapshot = run_synthetic([case], out, limits=limits, policies=(9,))
            rows = [json.loads(row) for row in (out/'landscapes.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0]['references']) & {'B00', 'B10', 'B01', 'B11'}, {'B00', 'B10', 'B01', 'B11'})
            self.assertTrue(rows[0]['controls'])
            self.assertFalse(rows[0]['arithmetic_failure'], rows[0].get('error'))
            self.assertIn('E', rows[0])
            self.assertIn('challenge_gains', rows[0])
            for ref in rows[0]['references'].values():
                self.assertIn('route_a', ref)
                self.assertIn('route_b', ref)
                self.assertIn('face_distances', ref)
                for attempt in ref['attempts']:
                    self.assertIn('settings', attempt)
                    self.assertIn('objective_call_cap', attempt)
                    self.assertGreaterEqual(attempt['cpu_seconds'], 0)
            self.assertLessEqual(snapshot['counters']['actual_evaluations'], 70000)
            if snapshot['counters'].get('shared') == limits.shared:
                self.assertEqual(snapshot['stop_reason'], 'shared_evaluation_ceiling')
            self.assertIn(status, ('ACCEPTED WITH EXPLICIT LIMITATIONS', 'INCONCLUSIVE', 'FAILED'))

    def test_known_failure_survives_incomplete_output(self):
        case = self.fixture(True)
        with tempfile.TemporaryDirectory() as temporary:
            with OutputStore(Path(temporary)/'synthetic') as store:
                phase = PhaseDriver([case], Budget(), store, expected_keys=[case.key], policies=(17,), synthetic=True).run()
                phase.phase_failures.append('independent arithmetic counterexample')
                self.assertEqual(phase.write_results(17, final_status='INCONCLUSIVE'), 'FAILED')

    def test_reporting_override_cannot_manufacture_acceptance(self):
        case = self.fixture(True)
        with tempfile.TemporaryDirectory() as temporary:
            with OutputStore(Path(temporary)/'synthetic') as store:
                phase = PhaseDriver([case], Budget(), store, expected_keys=[case.key], policies=(17,), synthetic=True).run()
                with self.assertRaises(ValueError): phase.write_results(17, final_status='ACCEPTED')
                self.assertFalse(phase.budget.mandatory_complete)

    def test_synthetic_identity_rejects_official_ids(self):
        from dataclasses import replace
        case = replace(self.fixture(), instance_id=0)
        with tempfile.TemporaryDirectory() as temporary:
            with OutputStore(Path(temporary)/'synthetic') as store:
                with self.assertRaises(ValueError): PhaseDriver([case], Budget(), store, expected_keys=[case.key], synthetic=True)

    def test_scheduled_work_precedes_triggered_reserve_wave(self):
        from dataclasses import replace
        from unittest.mock import Mock
        from struct2circuit.calibration.driver import CaseResult
        from struct2circuit.calibration.reference import Candidate, ReferenceResult
        cases = [replace(self.fixture(), audit=True), replace(self.fixture(), instance_id=1043, sentinel=True)]
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            with OutputStore(Path(temporary)/'toy') as store:
                phase = PhaseDriver(cases, Budget(), store, expected_keys=[c.key for c in cases], policies=(9,), synthetic=True)
                def routine(case):
                    events.append(('routine', case.instance_id))
                    search = Mock()
                    search.refine.side_effect = lambda ref, **kw: events.append(('audit', case.instance_id, ref.domain))
                    search.search.side_effect = lambda box, **kw: events.append(('sentinel', case.instance_id, box.name)) or ReferenceResult(box.name)
                    result = CaseResult(case, Mock(), search, references={name: ReferenceResult(name) for name in ('B00', 'B10', 'B01', 'B11')},
                                        production={9: Candidate(.3, .2, .1, 'production', 'toy')})
                    phase.results[case.canonical_key] = result
                with patch.object(phase, '_routine', side_effect=routine), patch.object(phase, '_settle', side_effect=lambda: events.append(('triggered',))):
                    phase.run()
        self.assertEqual(events[:2], [('routine', 1042), ('routine', 1043)])
        self.assertEqual([e[0] for e in events[2:]], ['audit']*4+['sentinel']*2+['triggered'])

    def test_domain_ordering_records_challenges_and_ambiguous_refinement(self):
        from dataclasses import replace
        from struct2circuit.calibration.driver import CaseResult
        from struct2circuit.calibration.reference import Candidate, ReferenceResult
        first = self.fixture()
        second = replace(first, graph_id=1, canonical_graph_id=1)
        for ambiguous in (False, True):
            with self.subTest(ambiguous=ambiguous), tempfile.TemporaryDirectory() as temporary:
                with OutputStore(Path(temporary)/'mock-records') as store:
                    phase = PhaseDriver([first, second], Budget(), store, expected_keys=[first.key, second.key], policies=(17,), synthetic=True)
                    for index, case in enumerate((first, second)):
                        values = dict(B00=(.110, .112 if ambiguous else .113)[index], B10=(.108, .105)[index],
                                      B01=(.110, .112 if ambiguous else .113)[index], B11=(.110, .113)[index],
                                      B22=(.106, .103)[index], Bsign=(.106, .103)[index])
                        refs = {name: ReferenceResult(name, pooled=value, complete=True) for name, value in values.items()}
                        phase.results[case.canonical_key] = CaseResult(case, None, None, references=refs,
                            production={17: Candidate(.3, .2, values['B00'], 'production', 'mock')}, required_challenges=True)
                    phase._paired_rows()
                    row = next(r for r in phase.pairs if r.get('domains') == 'B11/B22')
                    self.assertEqual(row['status'], 'reversal')
                    self.assertEqual((row['first_order'], row['second_order']), ('i', 'j'))
                    if ambiguous:
                        self.assertTrue(all('B00' in r.ranking_boxes for r in phase.results.values()))


if __name__ == '__main__': unittest.main()


class AdditionalSafetyTests(unittest.TestCase):
    def test_reserved_generation_requires_activation_before_generator(self):
        with tempfile.TemporaryDirectory() as temporary:
            authorization = Path(temporary)/'authorization.json'
            authorization.write_text(json.dumps(dict(calibration_execution_authorized=True, protocol_sha256=PROTOCOL_HASH, phases=['development'])))
            reserves = [s for s in instance_specs(True) if s.split == 'reserve']
            with patch('struct2circuit.problems.block_correlated_qubo', side_effect=AssertionError('Reserved generation prohibited')):
                with self.assertRaises(PermissionError): generate_cases(reserves, Budget(), authorization=authorization, phase='development')

    def test_final_manifest_reports_actual_synthetic_status(self):
        case = WorkflowTests().fixture(constant=True)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'toy'
            with healthy_resource_guard():
                status, _ = run_synthetic([case], output, limits=Limits())
            self.assertEqual(json.loads((output/'calibration_manifest.json').read_text())['status'], status)

    def test_declared_pair_cannot_be_silently_omitted(self):
        from dataclasses import replace
        case = WorkflowTests().fixture(constant=True)
        alias = replace(case, graph_id=3)
        with tempfile.TemporaryDirectory() as temporary:
            with OutputStore(Path(temporary)/'toy') as store:
                phase = PhaseDriver([case, alias], Budget(), store, expected_keys=[case.key, alias.key], synthetic=True).run()
                phase.pairs = []
                checks, expected = phase.checks(17)
                self.assertIn('pair:1042:0:0:3', expected)
                self.assertFalse(any(c.name == 'pair:1042:0:0:3' for c in checks))
                self.assertEqual(phase.status(17), 'INCONCLUSIVE')

    def test_lock_output_collision_precedes_any_official_generation(self):
        from struct2circuit.calibration.driver import run_official_phase
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            auth = root/'authorization.json'
            auth.write_text(json.dumps(dict(calibration_execution_authorized=True, protocol_sha256=PROTOCOL_HASH, phases=['development'])))
            lock = root/'occupied-lock.json'
            lock.write_bytes(b'existing evidence')
            protocol = Path(__file__).resolve().parents[1]/'paper/numerical_calibration_protocol_v1.md'
            with healthy_resource_guard(), patch('struct2circuit.calibration.driver.generate_cases', side_effect=AssertionError('No official generation')):
                with self.assertRaises(FileExistsError): run_official_phase(root/'out', phase='development', authorization=auth, protocol=protocol, lock_output=lock)
            self.assertFalse((root/'out').exists())
            self.assertEqual(lock.read_bytes(), b'existing evidence')


class FailurePersistenceRegressionTests(unittest.TestCase):
    def test_completed_oracle_failure_survives_next_budget_stop_and_serialization(self):
        case = WorkflowTests().fixture()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'failure'
            with OutputStore(output) as store:
                limits = Limits(essential=3,shared=1,total=3,landscape=3,optional=1,cpu_seconds=20)
                phase = PhaseDriver([case],Budget(limits),store,expected_keys=[case.key],policies=(17,),synthetic=True)
                original = case.simulator.state_dimensionless
                with patch.object(case.simulator,'state_dimensionless',side_effect=lambda *args:original(*args)*1.000001):
                    phase.run()
                result = phase.results[case.canonical_key]
                self.assertTrue(result.arithmetic_failure)
                self.assertEqual(result.error,'total_evaluation_ceiling')
                self.assertEqual(phase.write_results(17),'FAILED')
            row = json.loads((output/'landscapes.jsonl').read_text().splitlines()[0])
            failed = next(c for c in row['controls'] if c['status']=='fail')
            self.assertGreater(failed['errors']['native_normalization'],1e-10)
            self.assertEqual(failed['tolerance'],1e-10)
            self.assertIn('objective_identity',failed)
            persisted = [json.loads(line) for line in (output/'candidates.jsonl').read_text().splitlines()]
            self.assertTrue(any(r.get('control',{}).get('status')=='fail' for r in persisted))
            self.assertTrue((output/'decision.md').read_text().startswith('FAILED'))

    def test_interruption_before_oracle_completion_does_not_fabricate_failure(self):
        case = WorkflowTests().fixture()
        with tempfile.TemporaryDirectory() as temporary:
            with OutputStore(Path(temporary)/'interrupted') as store:
                phase = PhaseDriver([case],Budget(),store,expected_keys=[case.key],policies=(17,),synthetic=True)
                with patch('struct2circuit.calibration.driver.direct_oracle',side_effect=BudgetExhausted('unfinished oracle')):
                    phase.run()
                result = phase.results[case.canonical_key]
                self.assertEqual(result.controls,[])
                self.assertFalse(result.arithmetic_failure)
                self.assertEqual(phase.write_results(17),'INCONCLUSIVE')

    def test_selected_failure_is_latched_before_later_selected_oracle_interrupts(self):
        from struct2circuit.calibration.driver import CaseResult
        from struct2circuit.calibration.objective import Objective
        from struct2circuit.calibration.reference import Candidate
        case = WorkflowTests().fixture()
        with tempfile.TemporaryDirectory() as temporary:
            with OutputStore(Path(temporary)/'selected') as store:
                phase = PhaseDriver([case],Budget(),store,expected_keys=[case.key],policies=(17,),synthetic=True)
                obj = Objective(case.simulator,phase.budget,'selected')
                candidate = Candidate(.3,.4,obj.evaluate(.3,.4).gap,'production','mock')
                result = CaseResult(case,obj,None,production={17:candidate,9:candidate})
                failure = dict(kind='direct_expm',status='fail',point=candidate.point,errors=dict(probability=.0474))
                with patch('struct2circuit.calibration.driver.direct_oracle',side_effect=[failure,BudgetExhausted('next check')]):
                    with self.assertRaises(BudgetExhausted):phase._selected_oracles(result)
                self.assertTrue(result.arithmetic_failure)
                self.assertEqual(result.controls[0]['errors']['probability'],.0474)

    def test_invariance_completed_records_stream_before_interruption(self):
        from dataclasses import replace
        from struct2circuit.calibration.objective import Objective
        from struct2circuit.calibration.diagnostics import invariance_checks
        from struct2circuit.calibration.driver import CaseResult
        case = WorkflowTests().fixture()
        with tempfile.TemporaryDirectory() as temporary:
            with OutputStore(Path(temporary)/'invariant') as store:
                b = Budget(Limits(essential=2,shared=1,total=2,landscape=2,optional=1,cpu_seconds=20))
                obj = Objective(case.simulator,b,'invariance')
                phase = PhaseDriver([case],b,store,expected_keys=[case.key],policies=(17,),synthetic=True)
                result = CaseResult(case,obj,None);phase.results[case.canonical_key]=result
                original = case.simulator._evaluate_state
                with patch.object(case.simulator,'_evaluate_state',side_effect=lambda state:replace(original(state),normalized_gap=original(state).normalized_gap+.01)):
                    with self.assertRaises(BudgetExhausted):
                        invariance_checks(obj,[(.3,.4),(.5,.6)],record_callback=lambda record:phase._record_control(result,record))
                self.assertEqual(len(result.controls),1)
                self.assertEqual(result.controls[0]['kind'],'cost_units')
                self.assertGreater(result.controls[0]['error'],1e-10)
                self.assertTrue(result.arithmetic_failure)
                self.assertEqual(phase.write_results(17),'FAILED')


class CompletedDiagnosticRegressionTests(unittest.TestCase):
    def case(self, initialization='uniform', *, degenerate=False):
        from test_calibration_numerics import fixture
        from struct2circuit.calibration.driver import Case
        from struct2circuit.mixers import complete_mixer
        sim = fixture(initialization)
        if degenerate:
            sim = FeasibleSubspaceQAOA(sim.problem, complete_mixer(4), initialization=initialization)
        return Case(1042, 0, 0, 0, 0, sim, 64)

    def evidence(self, phase, output, *, interrupt_decision=False):
        if interrupt_decision:
            with patch.object(phase.store, 'decision', side_effect=OSError('decision interrupted')):
                with self.assertRaisesRegex(OSError, 'decision interrupted'):
                    phase.write_results(17)
            status = phase.status(17)
        else:
            status = phase.write_results(17)
        row = json.loads((output/'landscapes.jsonl').read_text())
        candidates = [json.loads(line) for line in (output/'candidates.jsonl').read_text().splitlines()]
        accounting = json.loads((output/'budget.json').read_text())
        return phase, status, row, candidates, accounting, (output/'decision.md').read_text()

    def cpu_run(self, *, corrupt, before=False, interrupt_decision=False):
        from struct2circuit.calibration.objective import Objective
        from scipy.linalg import expm
        clock = [0.]
        b = Budget(Limits(cpu_seconds=1), cpu_clock=lambda: clock[0], wall_clock=lambda: clock[0])
        case = self.case()
        point = (.3, .4)
        original = Objective.evaluate
        def evaluate(obj, u, beta, domain='B00', **kwargs):
            value = original(obj, u, beta, domain, **kwargs)
            if (u, beta) == point and domain == 'B11':
                clock[0] = 1.  # Numerical inputs have returned; terminal check must still run.
            return value
        def exponential(matrix):
            if before:
                clock[0] = 1.  # Native comparison input cannot be obtained after this stop.
            return np.eye(len(matrix)) if corrupt else expm(matrix)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'cpu'
            with OutputStore(output) as store:
                phase = PhaseDriver([case], b, store, expected_keys=[case.key], policies=(17,), synthetic=True)
                with patch('struct2circuit.calibration.driver._points', return_value=((point,), (), ())), \
                     patch.object(Objective, 'evaluate', new=evaluate), \
                     patch('struct2circuit.calibration.diagnostics.expm', side_effect=exponential):
                    phase.run()
                return self.evidence(phase, output, interrupt_decision=interrupt_decision)

    def test_cpu_stop_after_failed_comparison_preserves_serialized_failure(self):
        phase, status, row, candidates, accounting, decision = self.cpu_run(corrupt=True)
        self.assertEqual(status, 'FAILED')
        self.assertTrue(row['arithmetic_failure'])
        self.assertEqual(row['error'], 'cpu_ceiling')
        self.assertEqual(len(row['controls']), 1)
        control = row['controls'][0]
        self.assertEqual(control['status'], 'fail')
        self.assertGreater(control['errors']['probability'], .01)
        self.assertGreater(control['errors']['gap'], .01)
        self.assertEqual(control['tolerance'], 1e-10)
        self.assertIn('objective_identity', control)
        self.assertEqual(control['point'], [.3, .4])
        self.assertEqual([r['control'] for r in candidates if r.get('kind') == 'numerical_control'], [control])
        self.assertTrue(decision.startswith('FAILED'))
        self.assertIn('reference:B00', ' '.join(accounting['incomplete_required_work']))
        self.assertEqual(accounting['stop_reason'], 'cpu_ceiling')
        self.assertEqual(phase.budget.counts['actual_evaluations'], 3)
        with self.assertRaises(BudgetExhausted):
            phase.budget.request('stopped', 'B00')

    def test_cpu_stop_before_comparison_does_not_fabricate_failure(self):
        _, status, row, _, accounting, _ = self.cpu_run(corrupt=True, before=True)
        self.assertEqual(status, 'INCONCLUSIVE')
        self.assertFalse(row['arithmetic_failure'])
        self.assertEqual(row['controls'], [])
        self.assertEqual(accounting['counters']['actual_evaluations'], 2)

    def test_cpu_stop_after_passing_comparison_remains_inconclusive(self):
        _, status, row, _, accounting, _ = self.cpu_run(corrupt=False)
        self.assertEqual(status, 'INCONCLUSIVE')
        self.assertFalse(row['arithmetic_failure'])
        self.assertEqual([c['status'] for c in row['controls']], ['pass'])
        self.assertEqual(row['error'], 'cpu_ceiling')
        self.assertTrue(accounting['incomplete_required_work'])

    def test_final_decision_write_interruption_keeps_latched_and_flushed_failure(self):
        _, status, row, candidates, accounting, decision = self.cpu_run(corrupt=True, interrupt_decision=True)
        self.assertEqual(status, 'FAILED')
        self.assertTrue(row['arithmetic_failure'])
        self.assertEqual(row['controls'][0]['status'], 'fail')
        self.assertEqual(len([r for r in candidates if r.get('kind') == 'numerical_control']), 1)
        self.assertEqual(accounting['stop_reason'], 'cpu_ceiling')
        self.assertEqual(decision, '')  # An interrupted decision is not claimed to be finalized.

    def test_multiple_completed_checks_keep_failure_and_no_duplicate_records(self):
        from struct2circuit.calibration.diagnostics import direct_oracle, spectral_coordinates
        from struct2circuit.calibration.driver import CaseResult
        from struct2circuit.calibration.objective import Objective
        clock = [0.]
        b = Budget(Limits(cpu_seconds=1), cpu_clock=lambda: clock[0], wall_clock=lambda: clock[0])
        case = self.case()
        with tempfile.TemporaryDirectory() as temporary, OutputStore(Path(temporary)/'multiple') as store:
            phase = PhaseDriver([case], b, store, expected_keys=[case.key], policies=(17,), synthetic=True)
            obj = Objective(case.simulator, b, 'multiple')
            result = CaseResult(case, obj, None)
            phase.results[case.key] = result
            phase._run_control(result, direct_oracle, obj, (.3, .4))
            phase._run_control(result, spectral_coordinates, obj, (.4, .5))
            original = obj.evaluate
            def interrupted(*args, **kwargs):
                value = original(*args, **kwargs)
                clock[0] = 1.
                return value
            with patch.object(obj, 'evaluate', side_effect=interrupted), \
                 patch('struct2circuit.calibration.diagnostics.expm', side_effect=lambda m: np.eye(len(m))):
                with self.assertRaises(BudgetExhausted) as interruption:
                    phase._run_control(result, direct_oracle, obj, (.5, .6))
            result.error = str(interruption.exception)
            self.assertEqual([r['status'] for r in result.controls], ['pass', 'pass', 'fail'])
            self.assertEqual(phase.write_results(17), 'FAILED')
            self.assertEqual(b.counts['actual_evaluations'], 6)

    def test_spectral_comparison_is_recorded_before_cpu_stop(self):
        from struct2circuit.calibration.diagnostics import spectral_coordinates
        from struct2circuit.calibration.driver import CaseResult
        from struct2circuit.calibration.objective import Objective
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as temporary, \
                 OutputStore(Path(temporary)/'spectral') as store:
                clock = [0.]
                b = Budget(Limits(cpu_seconds=1), cpu_clock=lambda: clock[0], wall_clock=lambda: clock[0])
                case = self.case()
                obj = Objective(case.simulator, b, 'spectral')
                result = CaseResult(case, obj, None)
                phase = PhaseDriver([case], b, store, expected_keys=[case.key], policies=(17,), synthetic=True)
                phase.results[case.key] = result
                original = obj.evaluate
                def interrupted(*args, **kwargs):
                    value = original(*args, **kwargs)
                    from dataclasses import replace
                    clock[0] = 1.
                    return replace(value, gap=value.gap+.01) if corrupt else value
                with patch.object(obj, 'evaluate', side_effect=interrupted):
                    with self.assertRaises(BudgetExhausted) as interruption:
                        phase._run_control(result, spectral_coordinates, obj, (.3, .4))
                result.error = str(interruption.exception)
                self.assertEqual(len(result.controls), 1)
                self.assertEqual(result.arithmetic_failure, corrupt)
                self.assertEqual(phase.write_results(17), 'FAILED' if corrupt else 'INCONCLUSIVE')
                self.assertEqual(b.counts['actual_evaluations'], 2)

    def transport_run(self, *, cap, corrupt=False, degenerate=False, reconstruction_shift=0.):
        from dataclasses import replace
        from struct2circuit.calibration.objective import Objective
        from struct2circuit.calibration.reference import Candidate, ReferenceResult, ReferenceSearch
        case = self.case('mixer_low' if degenerate else 'uniform', degenerate=degenerate)
        point = (.3, .4)
        b = Budget(Limits(essential=cap, shared=1, total=cap, landscape=cap, optional=1, cpu_seconds=20))
        original = Objective.evaluate
        cyclic = []
        def evaluate(obj, *args, **kwargs):
            value = original(obj, *args, **kwargs)
            if obj.simulator.mixer.name == 'cyclic_transport':
                if not cyclic and corrupt:
                    value = replace(value, gap=value.gap+.01)
                elif cyclic and reconstruction_shift:
                    value = replace(value, gap=value.gap+reconstruction_shift)
                cyclic.append(value)
            return value
        def production(obj, policy):
            value = obj.evaluate(*point)
            return Candidate(*point, value.gap, 'production', 'mock'), value
        def search(searcher, domain, **kwargs):
            c = kwargs['production']
            return ReferenceResult(domain.name, candidates=[replace(c, route=r) for r in ('A', 'B')],
                pooled=c.value, route_a=c.value, route_b=c.value, r=0., faces_complete=True, complete=True)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'transport'
            with OutputStore(output) as store:
                phase = PhaseDriver([case], b, store, expected_keys=[case.key], policies=(17,), synthetic=True)
                with patch('struct2circuit.calibration.driver._points', return_value=((), (), (point,))), \
                     patch('struct2circuit.calibration.driver.production_search', side_effect=production), \
                     patch.object(ReferenceSearch, 'search', new=search), \
                     patch('struct2circuit.calibration.driver.spectral_coordinates', return_value=dict(kind='mock_prior', status='pass')), \
                     patch.object(phase, '_selected_oracles'), patch.object(Objective, 'evaluate', new=evaluate):
                    phase.run()
                return self.evidence(phase, output)

    def test_transport_failure_survives_rebuilt_budget_stop_and_serialization(self):
        phase, status, row, candidates, accounting, decision = self.transport_run(cap=5, corrupt=True)
        self.assertEqual(status, 'FAILED')
        self.assertTrue(row['arithmetic_failure'])
        transport = next(c for c in row['controls'] if c['kind'] == 'cyclic_transport')
        reconstruction = next(c for c in row['controls'] if c['kind'] == 'cyclic_reconstruction')
        self.assertAlmostEqual(transport['error'], .01)
        self.assertEqual(transport['status'], 'fail')
        self.assertIn('objective_identity', transport)
        self.assertNotIn('selection_status', transport)
        self.assertEqual(reconstruction['status'], 'unresolved')
        self.assertIsNone(reconstruction['reconstructed_gap_sensitivity'])
        self.assertEqual(reconstruction['reason'], 'total_evaluation_ceiling')
        self.assertEqual([r['control'] for r in candidates if r.get('kind') == 'numerical_control'], [transport])
        self.assertEqual(row['error'], 'total_evaluation_ceiling')
        self.assertTrue(accounting['incomplete_required_work'])
        self.assertTrue(decision.startswith('FAILED'))
        self.assertEqual(phase.budget.counts['actual_evaluations'], 5)

    def test_passing_transport_with_unfinished_reconstruction_is_inconclusive(self):
        _, status, row, _, _, _ = self.transport_run(cap=5)
        self.assertEqual(status, 'INCONCLUSIVE')
        self.assertFalse(row['arithmetic_failure'])
        self.assertEqual([c['status'] for c in row['controls'] if c['kind'].startswith('cyclic_')], ['pass', 'unresolved'])

    def test_interruption_before_transport_does_not_fabricate_comparison(self):
        _, status, row, _, _, _ = self.transport_run(cap=4, corrupt=True)
        self.assertEqual(status, 'INCONCLUSIVE')
        self.assertFalse(row['arithmetic_failure'])
        self.assertFalse(any(c['kind'].startswith('cyclic_') for c in row['controls']))

    def test_sufficient_budget_transport_controls_remain_separate(self):
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt):
                phase, status, row, _, _, _ = self.transport_run(cap=100, corrupt=corrupt)
                self.assertEqual(status, 'FAILED' if corrupt else 'ACCEPTED WITH EXPLICIT LIMITATIONS')
                self.assertEqual(row['arithmetic_failure'], corrupt)
                self.assertIsNone(row['error'])
                records = [c for c in row['controls'] if c['kind'].startswith('cyclic_')]
                self.assertEqual(len(records), 2)
                self.assertEqual(records[0]['status'], 'fail' if corrupt else 'pass')
                self.assertEqual(records[1]['status'], 'pass')
                self.assertEqual(records[1]['selection_status'], 'pass')
                self.assertEqual(phase.budget.counts['actual_evaluations'], 6)

    def test_degenerate_low_reconstruction_sensitivity_is_not_arithmetic_failure(self):
        from struct2circuit.calibration.diagnostics import spectral_diagnostics
        spectrum = spectral_diagnostics(self.case('mixer_low', degenerate=True).simulator)
        self.assertGreater(spectrum['extrema']['low']['multiplicity'], 1)
        self.assertEqual(spectrum['extrema']['low']['projection_branch'], 'lexicographic_fallback')
        # Inject sensitivity only into reconstruction; transport remains exact.
        _, _, row, _, _, _ = self.transport_run(cap=100, degenerate=True, reconstruction_shift=.01)
        transport = next(c for c in row['controls'] if c['kind'] == 'cyclic_transport')
        reconstruction = next(c for c in row['controls'] if c['kind'] == 'cyclic_reconstruction')
        self.assertEqual(transport['status'], 'pass')
        self.assertEqual(reconstruction['status'], 'pass')
        self.assertGreater(reconstruction['reconstructed_gap_sensitivity'], 1e-3)
        self.assertEqual(reconstruction['selection_status'], 'fail')
        self.assertFalse(row['arithmetic_failure'])


class NestedPoolRegressionTests(unittest.TestCase):
    def run_improvement(self, step):
        from struct2circuit.calibration.driver import CaseResult
        from struct2circuit.calibration.objective import Objective
        from struct2circuit.calibration.reference import Candidate,ReferenceResult,ReferenceSearch
        from struct2circuit.calibration.config import BY_NAME
        case = WorkflowTests().fixture()
        # Generic native landscape, not the complement-symmetric workflow toy.
        Q = np.diag([.1,.7,-.5,1.]);Q[0,2]=Q[2,0]=.23
        sim = FeasibleSubspaceQAOA(CardinalityQUBO(Q,np.array([-.4,.1,.3,-.2]),2),ring_mixer(4))
        from dataclasses import replace
        case = replace(case,simulator=sim)
        with tempfile.TemporaryDirectory() as temporary:
            with OutputStore(Path(temporary)/'pool') as store:
                b = Budget();obj = Objective(sim,b,'nested');search = ReferenceSearch(obj,64,case.canonical_key)
                p,q = (.5,.5),(.5+step,.5)
                old,new = obj.evaluate(*p).gap,obj.evaluate(*q).gap
                if new>old:p,q=q,p;old,new=new,old
                phase = PhaseDriver([case],b,store,expected_keys=[case.key],policies=(17,),synthetic=True)
                candidates = [Candidate(*p,old,r,'old',objective_identity=obj.identity) for r in ('A','B')]
                refs = {n:ReferenceResult(n,candidates=list(candidates),route_a=old,route_b=old,pooled=old,
                    r=0.,faces_complete=True,complete=True) for n in ('B00','B10','B01','B11','B22','Bsign')}
                for n in ('B22','Bsign'):
                    refs[n].refined=refs[n].full_bundle=True;refs[n].coarse=old;refs[n].f=0.
                result = CaseResult(case,obj,search,references=refs,production={17:candidates[0]},required_challenges=True)
                phase.results[case.canonical_key]=result
                def refine(ref,**kw):
                    ref.coarse=ref.pooled;ref.refined=ref.full_bundle=True
                    ref.candidates.extend(Candidate(*q,new,r,'fine',objective_identity=obj.identity) for r in ('A','B'))
                    search._pool(ref,BY_NAME[ref.domain],());return ref
                reasons = lambda r: {n:(('test_trigger',) if n=='B11' and not refs[n].refined else ()) for n in refs}
                with patch.object(search,'refine',side_effect=refine), patch.object(phase,'_reference_checks',side_effect=reasons):
                    phase._settle()
                for n in ('B22','Bsign'):
                    ref=refs[n]
                    self.assertLessEqual(ref.pooled,refs['B11'].pooled)
                    imported=[c for c in ref.candidates if c.point==q]
                    self.assertTrue(imported)
                    self.assertTrue(all(c.route=='pool' for c in imported))
                    self.assertEqual(ref.route_a,old);self.assertEqual(ref.route_b,old)
                    self.assertAlmostEqual(ref.r,old-new);self.assertAlmostEqual(ref.f,old-new)
                before=b.counts['actual_evaluations'];sizes={n:len(r.candidates) for n,r in refs.items()}
                phase._pool_references(result)
                self.assertEqual(b.counts['actual_evaluations'],before)
                self.assertEqual(sizes,{n:len(r.candidates) for n,r in refs.items()})
                phase.write_results(17)
            row=json.loads((Path(temporary)/'pool/landscapes.jsonl').read_text())
            return old-new,refs,row

    def test_late_native_candidate_reaches_both_completed_challenges(self):
        improvement,refs,row=self.run_improvement(.00001)
        self.assertAlmostEqual(improvement,2.3802616467838433e-6)
        self.assertTrue(refs['B22'].complete and refs['Bsign'].complete)
        self.assertAlmostEqual(row['challenge_gains']['step'],0.)
        self.assertAlmostEqual(row['challenge_gains']['signed'],0.)

    def test_imported_winner_cannot_manufacture_independent_agreement(self):
        improvement,refs,row=self.run_improvement(.001)
        self.assertGreater(improvement,1e-5)
        self.assertFalse(refs['B22'].complete or refs['Bsign'].complete)
        self.assertIsNone(row['challenge_gains']['step'])
        self.assertIsNone(row['challenge_gains']['signed'])


class PendingTriggerRegressionTests(unittest.TestCase):
    def make_phase(self,store,*,late_unresolved=True,late_deficit=.0003):
        from dataclasses import replace
        from types import SimpleNamespace
        from unittest.mock import Mock
        from struct2circuit.calibration.driver import CaseResult
        from struct2circuit.calibration.reference import Candidate,ReferenceResult
        first=WorkflowTests().fixture();second=replace(first,graph_id=1,canonical_graph_id=1)
        phase=PhaseDriver([second,first],Budget(),store,expected_keys=[first.key,second.key],policies=(17,),synthetic=True)
        events=[]
        for i,case in enumerate((first,second)):
            value=(.2,.2994)[i];candidate=Candidate(1.,1.,value,'A','mock')
            refs={n:ReferenceResult(n,candidates=[candidate],pooled=value,route_a=value,route_b=value,
                r=0.,faces_complete=True,complete=True) for n in ('B00','B10','B01','B11')}
            if i and late_unresolved:refs['B00'].complete=False;refs['B00'].r=.0003
            search=Mock()
            def refine(ref,index=i,**kw):
                events.append((index,ref.domain));ref.coarse=ref.pooled;ref.f=0.;ref.r=0.
                ref.refined=ref.full_bundle=ref.complete=True;return ref
            search.refine.side_effect=refine
            phase.results[case.canonical_key]=CaseResult(case,SimpleNamespace(account='essential'),search,references=refs,
                production={17:replace(candidate,value=value+(late_deficit if i else 0.))})
        return phase,events,first,second

    def test_late_pair_trigger_revisits_earlier_member_exactly_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            with OutputStore(Path(temporary)/'pending') as store:
                phase,events,first,second=self.make_phase(store)
                with patch.object(phase,'_selected_oracles'):
                    phase._settle()
                self.assertEqual(events,[(1,'B00'),(0,'B00')])
                self.assertTrue(all(r.references['B00'].full_bundle for r in phase.results.values()))
                self.assertFalse(any(phase.pending_work.values()))
                self.assertEqual(phase.status(17),'FAILED')  # Resolved E/K counterexample is retained.
                pair=next(p for p in phase.pairs if p['policy']==17)
                self.assertAlmostEqual(pair['K'],.0003)
                with patch.object(phase,'_selected_oracles'):phase._settle()
                self.assertEqual(events,[(1,'B00'),(0,'B00')])

    def test_budget_stop_with_unresolved_tasks_never_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            with OutputStore(Path(temporary)/'stop') as store:
                phase,events,first,second=self.make_phase(store,late_deficit=0.)
                for result in phase.results.values():
                    result.references['B00'].r=.01;result.references['B00'].complete=False
                    def stop(*args,**kw):
                        phase.budget.stop_reason='cpu_ceiling';raise BudgetExhausted('cpu_ceiling')
                    result.search.refine.side_effect=stop
                with patch.object(phase,'_selected_oracles'):phase._settle()
                self.assertEqual(phase.status(17),'INCONCLUSIVE')
                phase.results[first.canonical_key].arithmetic_failure=True
                self.assertEqual(phase.status(17),'FAILED')
                self.assertLessEqual(phase.budget.counts['actual_evaluations'],phase.budget.limits.total)

    def test_backend_without_progress_has_bounded_termination_and_unresolved_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            with OutputStore(Path(temporary)/'bounded') as store:
                phase,events,first,second=self.make_phase(store,late_deficit=0.)
                result=phase.results[second.canonical_key]
                result.search.refine.side_effect=None
                with patch.object(phase,'_selected_oracles'):phase._settle()
                self.assertTrue(phase.pending_work[second.canonical_key])
                self.assertEqual(result.search.refine.call_count,1)
                self.assertEqual(phase.status(17),'INCONCLUSIVE')


class DevelopmentFinalizationRegressionTests(unittest.TestCase):
    def test_entry_fixtures_work_without_cgroups_while_real_guard_rejects(self):
        original_is_file = Path.is_file
        def no_cgroups(path):
            return False if str(path).startswith('/sys/fs/cgroup/') else original_is_file(path)
        with tempfile.TemporaryDirectory() as temporary, patch.object(Path, 'is_file', new=no_cgroups):
            root = Path(temporary)
            budget = Budget()
            with self.assertRaisesRegex(ResourceExhausted, 'Readable cgroup memory limits are required'):
                ResourceGuard(root).check(budget, force=True)
            self.assertEqual(budget.stop_reason, 'resource_information_unavailable')
            with healthy_resource_guard():
                status, snapshot = run_synthetic([WorkflowTests().fixture(constant=True)],
                    root/'synthetic', limits=Limits())
            self.assertEqual(status, 'INCONCLUSIVE')
            self.assertEqual(snapshot['counters'].get('actual_evaluations', 0), 0)
            self.assertEqual(self.official_mock(root), 'ACCEPTED WITH EXPLICIT LIMITATIONS')
            # The test-local injection must not leak into subsequent admission.
            with self.assertRaises(ResourceExhausted): ResourceGuard(root).check(force=True)

    def official_mock(self, root, *, interruption=None):
        """Exercise the real entry's finalization; all instance generation is mocked."""
        from types import SimpleNamespace
        from struct2circuit.calibration.driver import run_official_phase
        from struct2circuit.calibration.decisions import Check
        protocol=Path(__file__).resolve().parents[1]/'paper/numerical_calibration_protocol_v1.md'
        auth=root/'auth.json'
        auth.write_text(json.dumps(dict(calibration_execution_authorized=True,protocol_sha256=PROTOCOL_HASH,
                                        phases=['development','validation'])))
        sim=WorkflowTests().fixture().simulator
        def mock_generate(specs,budget,**kw):
            return tuple(SimpleNamespace(instance_id=s.instance_id,simulator=sim) for s in specs)
        class MockCompletedPhase:
            def __init__(self,cases,budget,store,expected_keys,**kw):
                self.cases=cases;self.budget=budget;self.store=store;self.expected_keys=expected_keys;self.results={};self.phase_failures=[]
            def run(self):return self
            def checks(self,n):return [Check('mock:E','pass')],['mock:E']
            def status(self,n):return 'ACCEPTED WITH EXPLICIT LIMITATIONS'
            def write_results(self,n,**kw):
                self.store.row('candidates.jsonl',dict(mock=True))
                for key in self.expected_keys:
                    self.store.row('graphs.jsonl',dict(landscape=key,mock=True))
                    self.store.row('landscapes.jsonl',dict(landscape=key,status='observed',arithmetic_failure=False,
                        references={b:dict(complete=True) for b in ('B00','B10','B01','B11')},mock=True))
                self.store.json('budget.json',self.budget.snapshot()|dict(incomplete_required_work=[]))
                if interruption=='before_decision':raise OSError('before decision')
                self.store.decision('INCONCLUSIVE\n\nPhase numerical status: ACCEPTED WITH EXPLICIT LIMITATIONS.\n'
                    'Policy: P17 on B00 only.\nScope: bounded_B00_native_common_physical_beta. Validation completed: False. Synthetic check: False.')
                return self.status(n)
        original_decision=OutputStore.decision
        original_json=OutputStore.json
        def decision(store,value):
            if interruption=='during_decision':
                store.handles['decision.md'].write('partial');store.handles['decision.md'].flush()
                raise OSError('during decision')
            return original_decision(store,value)
        def json_write(store,name,value,**kw):
            if name=='calibration_manifest.json' and kw.get('update_owned') and interruption=='before_manifest':
                raise OSError('after decision before manifest')
            return original_json(store,name,value,**kw)
        with healthy_resource_guard(), patch('struct2circuit.calibration.driver.generate_cases',side_effect=mock_generate), \
             patch('struct2circuit.calibration.driver.PhaseDriver',new=MockCompletedPhase), \
             patch.object(OutputStore,'decision',new=decision),patch.object(OutputStore,'json',new=json_write):
            return run_official_phase(root/'development',phase='development',authorization=auth,
                                      protocol=protocol,lock_output=root/'lock.json')

    def assert_restart_blocked(self,root):
        from struct2circuit.calibration.driver import run_official_phase
        protocol=Path(__file__).resolve().parents[1]/'paper/numerical_calibration_protocol_v1.md'
        with healthy_resource_guard(), patch('struct2circuit.calibration.driver.generate_cases',side_effect=AssertionError('No approved generator')) as generate:
            with self.assertRaises((ValueError,FileNotFoundError,json.JSONDecodeError)):
                run_official_phase(root/'validation',phase='validation',authorization=root/'auth.json',protocol=protocol,lock=root/'lock.json')
            self.assertFalse(generate.called)

    def test_result_writing_failure_cannot_publish_usable_lock(self):
        for interruption in ('before_decision','during_decision','before_manifest'):
            with self.subTest(interruption=interruption),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                with self.assertRaises(OSError):self.official_mock(root,interruption=interruption)
                self.assertFalse((root/'lock.json').exists())
                self.assert_restart_blocked(root)

    def test_before_atomic_publication_leaves_no_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch('struct2circuit.calibration.storage.os.link',side_effect=OSError('before publication')):
                with self.assertRaises(OSError):self.official_mock(root)
            self.assertFalse((root/'lock.json').exists())
            self.assertFalse(list(root.glob('lock.json.*.tmp')))
            self.assert_restart_blocked(root)

    def test_after_atomic_publication_contains_only_complete_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            # Simulate process interruption after publication, before returning a gate.
            with patch('struct2circuit.calibration.storage.ValidationGate',side_effect=OSError('after publication')):
                with self.assertRaises(OSError):self.official_mock(root)
            self.assertTrue((root/'lock.json').exists())
            gate=ValidationGate(root/'lock.json')
            self.assertTrue(gate.payload['development_evidence']['sha256'])
            gate.begin_generation(17)
            with self.assertRaises(FileExistsError):ValidationGate(root/'lock.json').begin_generation(17)

    def test_finalized_entry_reaches_mock_generation_after_restart_only_once(self):
        from struct2circuit.calibration.driver import run_official_phase
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.assertEqual(self.official_mock(root),'ACCEPTED WITH EXPLICIT LIMITATIONS')
            class ReachedMockGenerator(Exception):pass
            def generation(specs,budget,**kw):
                kw['validation_gate'].begin_generation(kw['policy']);raise ReachedMockGenerator()
            protocol=Path(__file__).resolve().parents[1]/'paper/numerical_calibration_protocol_v1.md'
            with healthy_resource_guard(), patch('struct2circuit.calibration.driver.generate_cases',side_effect=generation):
                with self.assertRaises(ReachedMockGenerator):
                    run_official_phase(root/'validation',phase='validation',authorization=root/'auth.json',protocol=protocol,lock=root/'lock.json')
            with healthy_resource_guard(), patch('struct2circuit.calibration.driver.generate_cases',side_effect=AssertionError('Second validation')) as generate:
                with self.assertRaises(FileExistsError):
                    run_official_phase(root/'second',phase='validation',authorization=root/'auth.json',protocol=protocol,lock=root/'lock.json')
                self.assertFalse(generate.called)

    def test_changed_missing_or_running_evidence_is_rejected_at_restart(self):
        for name,change in (('decision.md','empty'),('decision.md','partial'),('calibration_manifest.json','running'),
                            ('calibration_manifest.json','policy'),('graphs.jsonl','missing'),('budget.json','partial')):
            with self.subTest(name=name,change=change),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);self.official_mock(root)
                path=root/'development'/name
                if change=='missing':path.unlink()
                elif change in ('empty','partial'):path.write_text('' if change=='empty' else 'partial')
                else:
                    document=json.loads(path.read_text());document['status' if change=='running' else 'selected_policy']='RUNNING' if change=='running' else 9
                    path.write_text(canonical_json(document))
                self.assert_restart_blocked(root)

    def test_evidence_is_rechecked_after_gate_construction_before_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.official_mock(root);gate=ValidationGate(root/'lock.json')
            (root/'development/decision.md').write_text('partial')
            with self.assertRaises(ValueError):gate.begin_generation(17)
            self.assertFalse((root/'lock.json.validation-started').exists())

    def test_legacy_or_truncated_lock_cannot_bypass_finalized_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.official_mock(root)
            lock=root/'lock.json';document=json.loads(lock.read_text())
            document['payload'].pop('development_evidence')
            document['sha256']=sha256(canonical_json(document['payload']).encode()).hexdigest()
            lock.write_text(canonical_json(document))
            self.assert_restart_blocked(root)
            lock.write_text('{"payload":')
            self.assert_restart_blocked(root)

    def test_claimed_finalization_with_inconsistent_decision_or_coverage_is_rejected(self):
        from struct2circuit.calibration.config import SCOPE
        for mutation in ('policy','coverage','running'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);b=Budget();b.mandatory_complete=True
                evidence=finalized_mock_development(root/'evidence',b)
                if mutation=='policy':(evidence/'decision.md').write_text('wrong policy')
                elif mutation=='coverage':(evidence/'landscapes.jsonl').write_text('{}\n')
                else:
                    m=json.loads((evidence/'calibration_manifest.json').read_text());m['status']='RUNNING'
                    (evidence/'calibration_manifest.json').write_text(canonical_json(m))
                with self.assertRaises(ValueError):
                    create_validation_lock(root/'lock.json',policy=17,scope=SCOPE,reserves=False,budget=b,
                        development_status='ACCEPTED WITH EXPLICIT LIMITATIONS',development_output=evidence)
                self.assertFalse((root/'lock.json').exists())

    def test_empty_pair_csv_has_the_existing_schema_header(self):
        import csv
        from struct2circuit.calibration.storage import PAIR_FIELDS
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with OutputStore(root/'out'):pass
            with (root/'out/paired_contrasts.csv').open() as handle:
                reader=csv.reader(handle);self.assertEqual(next(reader),list(PAIR_FIELDS));self.assertEqual(list(reader),[])

    def test_partial_temporary_lock_write_never_publishes(self):
        from struct2circuit.calibration import storage
        original=storage.tempfile.NamedTemporaryFile
        class PartialWrite:
            def __init__(self,*args,**kwargs):self.file=original(*args,**kwargs)
            def __enter__(self):self.file.__enter__();self.name=self.file.name;return self
            def __exit__(self,*args):return self.file.__exit__(*args)
            def write(self,text):
                self.file.write(text[:len(text)//2]);self.file.flush();raise OSError('partial lock write')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch.object(storage.tempfile,'NamedTemporaryFile',new=PartialWrite):
                with self.assertRaises(OSError):self.official_mock(root)
            self.assertFalse((root/'lock.json').exists())
            self.assertFalse(list(root.glob('lock.json.*.tmp')))
            self.assert_restart_blocked(root)
