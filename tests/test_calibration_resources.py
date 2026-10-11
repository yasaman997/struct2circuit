"""Bounded resource/lifecycle regressions; never generate official instances."""
from pathlib import Path
from dataclasses import replace
from contextlib import ExitStack
from unittest.mock import patch
import gc
import hashlib
import io
import json
import sys
import tempfile
import tracemalloc
import unittest
import weakref

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import test_calibration_safety as safety
from struct2circuit.calibration.budget import Budget, BudgetExhausted, Limits
from struct2circuit.calibration.driver import CaseResult, PhaseDriver, run_synthetic
from struct2circuit.calibration.objective import Objective
from struct2circuit.calibration.reference import Candidate, ReferenceResult, ReferenceSearch
from struct2circuit.calibration.decisions import band, DISTORTION
from struct2circuit.calibration.resources import GIB, MIB, ResourceGuard, ResourceExhausted, resource_sample
from struct2circuit.calibration.storage import (OutputStore, canonical_chunks, canonical_hash,
                                               canonical_json, write_canonical)


def key_budget(count=20):
    b = Budget(cpu_clock=lambda: 0., wall_clock=lambda: 0.)
    for i in range(count):
        key = ('synthetic-identity-'+str(i%2), i.to_bytes(16, 'big'))
        b.request('toy', 'B00'); b.charge(key, 'toy', 'B00')
    return b


def healthy(paths):
    return 32*GIB, 30*GIB, 64*MIB, 10*GIB


class ResourceLifecycleTests(unittest.TestCase):
    def test_release_preserves_values_gradients_evidence_and_all_accounting(self):
        case = safety.WorkflowTests().fixture()
        b = Budget(cpu_clock=lambda: 0., wall_clock=lambda: 0.)
        obj = Objective(case.simulator, b, 'toy')
        entry = obj.evaluate(.7, .4)
        record = entry.record(); gradient = obj.gradient((.7, .4), 'B00')
        np.testing.assert_array_equal(gradient, obj.gradient((.7, .4), 'B00'))
        snapshot = b.snapshot(); state = weakref.ref(entry.state)
        del entry
        obj.release_cache(); gc.collect()
        self.assertIsNone(state()); self.assertIsNone(obj.last); self.assertEqual(obj.cache, {})
        self.assertEqual(snapshot, b.snapshot()); self.assertTrue(np.isfinite(record['normalized_gap']))
        with self.assertRaisesRegex(RuntimeError, 'terminal cache release'): obj.evaluate(.7, .4)
        self.assertEqual(snapshot, b.snapshot())

    def test_terminal_reporting_preserves_references_and_rejects_resume_or_policy(self):
        case = safety.WorkflowTests().fixture()
        limits = Limits(essential=2000, shared=1, total=2001, landscape=2000, optional=1, cpu_seconds=20)
        with tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
            b = Budget(limits)
            phase = PhaseDriver([case], b, store, expected_keys=[case.key], policies=(9,), synthetic=True).run()
            result = phase.results[case.canonical_key]
            refs = {name: [(c.point, c.value, c.route, c.source) for c in r.candidates] for name,r in result.references.items()}
            self.assertTrue(refs); self.assertTrue(result.objective.cache)
            counts, unique = dict(b.counts), set(b.unique)
            status = phase.write_results(9)
            self.assertEqual(status, 'INCONCLUSIVE')
            self.assertEqual(counts, dict(b.counts)); self.assertEqual(unique, b.unique)
            self.assertFalse(result.objective.cache); self.assertIsNone(result.objective.last)
            self.assertEqual(refs, {name:[(c.point,c.value,c.route,c.source) for c in r.candidates] for name,r in result.references.items()})
            with self.assertRaises(RuntimeError): phase.add_policy(33)
            with self.assertRaises(RuntimeError): phase.run()

    def test_resource_stop_preserves_completed_arithmetic_failure(self):
        case = safety.WorkflowTests().fixture(); b = Budget()
        with tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
            phase = PhaseDriver([case], b, store, expected_keys=[case.key], policies=(9,), synthetic=True)
            obj = Objective(case.simulator,b,'toy'); obj.evaluate(.2,.3)
            result = CaseResult(case,obj,ReferenceSearch(obj,64,case.canonical_key))
            phase.results[case.canonical_key] = result
            phase._record_control(result, dict(kind='direct_expm',status='fail',errors=dict(gap=.1)))
            samples = iter([(32*GIB, 30*GIB, 24*GIB, 10*GIB), healthy(())])
            b.resource_guard = ResourceGuard(store.directory, sample=lambda p:next(samples))
            with self.assertRaises(ResourceExhausted): b.check_time()
            result.error = b.stop_reason
            self.assertEqual(phase.write_results(9,final_status='INCONCLUSIVE'), 'FAILED')
            self.assertFalse(b.mandatory_complete)
            self.assertIn('"status":"fail"', (store.directory/'candidates.jsonl').read_text())

    def test_emergency_reporting_preserves_failure_without_full_snapshot(self):
        from struct2circuit.calibration.driver import _report_resource_stop
        case=safety.WorkflowTests().fixture();b=Budget()
        with tempfile.TemporaryDirectory() as t,OutputStore(Path(t)/'out') as store:
            phase=PhaseDriver([case],b,store,expected_keys=[case.key],policies=(9,),synthetic=True)
            obj=Objective(case.simulator,b,'toy');obj.evaluate(.2,.3)
            result=CaseResult(case,obj,ReferenceSearch(obj,64,case.canonical_key))
            phase.results[case.canonical_key]=result
            phase._record_control(result,dict(kind='direct_expm',status='fail'))
            b.stop_reason='resource_disk_headroom';keys=set(b.unique);counts=dict(b.counts)
            self.assertEqual(_report_resource_stop(store,b,ResourceExhausted('no headroom'),phase,'synthetic',policy=9),'FAILED')
            self.assertEqual(keys,b.unique);self.assertEqual(counts,dict(b.counts));self.assertFalse(obj.cache)
            self.assertEqual(json.loads((store.directory/'calibration_manifest.json').read_text())['status'],'FAILED')
            self.assertEqual((store.directory/'budget.json').stat().st_size,0)


class EmergencyFailurePrecedenceTests(unittest.TestCase):
    def phase(self, store, kind='none', policy=17):
        first = safety.WorkflowTests().fixture()
        cases = [first]
        if kind in ('K', 'combined'):
            cases.append(replace(first, graph_id=1, canonical_graph_id=1))
        b = key_budget(2)
        phase = PhaseDriver(cases, b, store, expected_keys=[c.key for c in cases],
                            policies=(9, 17, 33), synthetic=True)
        for case in cases:
            obj = Objective(case.simulator, b, str(case.canonical_key))
            candidate = Candidate(.2, .3, .2, 'A', 'mock_completed', objective_identity=obj.identity)
            refs = {name: ReferenceResult(name, candidates=[candidate], pooled=.2,
                route_a=.2, route_b=.2, r=0., faces_complete=True, complete=True)
                for name in ('B00', 'B10', 'B01', 'B11')}
            result = CaseResult(case, obj, ReferenceSearch(obj, 64, case.canonical_key),
                references=refs, production={n: replace(candidate, route='production') for n in (9, 17, 33)})
            phase.results[case.canonical_key] = result
        result = phase.results[first.canonical_key]
        if kind in ('E', 'combined'):
            result.production[policy] = replace(result.production[policy], value=.201)
        if kind == 'arithmetic':
            phase._record_control(result, dict(kind='mock_completed_oracle', status='fail'))
        if kind == 'unresolved':
            result.references['B00'].complete = False
            result.production.pop(policy)
        if kind == 'optional':
            result.references['B10'].pooled = .1  # broader claims are nonessential here
        if kind in ('K', 'combined', 'unresolved'):
            # Mock a completed pair classification independently of the E fixture.
            # Emergency reporting must inspect this record, never recalculate it.
            phase.pairs.append(dict(instance_id=first.instance_id, initialization_id=0,
                i=0, j=1, policy=policy, K=None if kind == 'unresolved' else .0003,
                status='unresolved' if kind == 'unresolved' else band(.0003, DISTORTION)))
        return phase

    def report(self, phase, store, *, policy=17, disk=False):
        from struct2circuit.calibration.driver import _report_resource_stop
        b = phase.budget
        b.resource_guard = ResourceGuard(store.directory, sample=lambda p:
            (32*GIB, 30*GIB, 64*MIB, 100*MIB) if disk else (32*GIB, 8*GIB, 24*GIB, 10*GIB))
        with self.assertRaises(ResourceExhausted) as caught:
            b.resource_guard.check(b, force=True)
        counts, keys = dict(b.counts), set(b.unique)
        with ExitStack() as stack:
            stack.enter_context(patch.object(b, 'snapshot', side_effect=AssertionError('no snapshot')))
            stack.enter_context(patch.object(Objective, 'evaluate', side_effect=AssertionError('no evaluation')))
            for name in ('status', '_paired_rows', '_settle', 'run', 'add_policy'):
                stack.enter_context(patch.object(phase, name, side_effect=AssertionError('no numerical/lock work')))
            status = _report_resource_stop(store, b, caught.exception, phase, 'synthetic', policy=policy)
        self.assertEqual(counts, dict(b.counts)); self.assertEqual(keys, b.unique)
        self.assertFalse(b.mandatory_complete)
        self.assertTrue(all(r.objective.released for r in phase.results.values()))
        manifest = json.loads((store.directory/'calibration_manifest.json').read_text())
        decision = (store.directory/'decision.md').read_text()
        self.assertEqual(status, manifest['status']); self.assertEqual(status, decision.splitlines()[0])
        self.assertFalse(manifest['reporting_complete']); self.assertFalse(manifest['development_finalized'])
        self.assertEqual((store.directory/'budget.json').stat().st_size, 0)
        self.assertIn(b.stop_reason, ('resource_memory_headroom', 'resource_disk_headroom'))
        self.assertEqual(manifest['stop_reason'], b.stop_reason)
        self.assertIn(str(caught.exception), decision)
        return status, manifest, decision

    def test_completed_E_failure_survives_memory_and_disk_stop(self):
        for disk in (False, True):
            with self.subTest(disk=disk), tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
                phase = self.phase(store, 'E')
                self.assertEqual(phase.status(17), 'FAILED')
                status, manifest, decision = self.report(phase, store, disk=disk)
                self.assertEqual(status, 'FAILED')
                self.assertIn('1042:0:0:E', manifest['failed_checks'])
                self.assertIn('1042:0:0:E', decision)

    def test_completed_K_failure_survives_without_E_failure(self):
        with tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
            phase = self.phase(store, 'K')
            self.assertTrue(all(c.status == 'pass' for c in phase.checks(17)[0] if c.name.endswith(':E')))
            self.assertEqual(phase.status(17), 'FAILED')
            status, manifest, decision = self.report(phase, store)
            self.assertEqual(status, 'FAILED')
            self.assertEqual(manifest['failed_checks'], ['pair:1042:0:0:1'])
            self.assertEqual(manifest['selected_policy'], 17)
            self.assertIn('pair:1042:0:0:1', decision)

    def test_combined_completed_failures_retain_all_identifiers(self):
        with tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
            status, manifest, decision = self.report(self.phase(store, 'combined'), store)
            self.assertEqual(status, 'FAILED')
            self.assertEqual(set(manifest['failed_checks']), {'1042:0:0:E', 'pair:1042:0:0:1'})
            self.assertTrue(all(name in decision for name in manifest['failed_checks']))

    def test_arithmetic_latch_survives(self):
        with tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
            status, manifest, _ = self.report(self.phase(store, 'arithmetic'), store)
            self.assertEqual(status, 'FAILED'); self.assertIn('1042:0:0:arithmetic', manifest['failed_checks'])

    def test_neutral_unresolved_and_optional_controls_do_not_fabricate_failure(self):
        for kind in ('none', 'unresolved', 'optional'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
                phase = self.phase(store, kind)
                if kind == 'unresolved':
                    self.assertTrue(all(c.status == 'unresolved' for c in phase.checks(17)[0]
                                        if c.name.endswith(':E') or c.name.startswith('pair:')))
                status, manifest, _ = self.report(phase, store)
                self.assertEqual(status, 'INCONCLUSIVE'); self.assertEqual(manifest['failed_checks'], [])

    def test_explicit_policy_controls_E_and_K_precedence(self):
        for policy in (9, 17, 33):
            for kind in ('E', 'K'):
                with self.subTest(policy=policy, kind=kind), tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
                    phase = self.phase(store, kind, policy)
                    status, manifest, decision = self.report(phase, store, policy=policy)
                    self.assertEqual(status, 'FAILED'); self.assertEqual(manifest['selected_policy'], policy)
                    self.assertIn(f'Policy: P{policy}.', decision)

    def test_other_essential_claims_use_existing_classification(self):
        with tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
            phase = self.phase(store, 'optional'); phase.broader_essential = True
            status, manifest, _ = self.report(phase, store)
            self.assertEqual(status, 'FAILED')
            self.assertIn('1042:0:0:gain:B10', manifest['failed_checks'])

    def test_unselected_policy_failure_does_not_substitute_for_selected_policy(self):
        for selected in (9, 17, 33):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
                other = next(p for p in (9, 17, 33) if p != selected)
                phase = self.phase(store, 'combined', other)
                self.assertEqual(phase.status(other), 'FAILED')
                status, manifest, _ = self.report(phase, store, policy=selected)
                self.assertEqual(status, 'INCONCLUSIVE'); self.assertEqual(manifest['failed_checks'], [])
                self.assertEqual(manifest['selected_policy'], selected)

    def test_failed_check_list_is_capped_without_losing_failure_status(self):
        with tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
            phase = self.phase(store); phase.phase_failures = ['mock-'+str(i) for i in range(100)]
            status, manifest, _ = self.report(phase, store)
            self.assertEqual(status, 'FAILED'); self.assertEqual(len(manifest['failed_checks']), 64)
            self.assertTrue(manifest['failed_checks_truncated'])

    def test_memory_error_preserves_latched_failure_and_reports_inspection_limit(self):
        for kind, expected in (('none', 'INCONCLUSIVE'), ('arithmetic', 'FAILED')):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
                phase = self.phase(store, kind)
                with patch.object(phase, 'checks', side_effect=MemoryError('synthetic inspection failure')):
                    status, manifest, decision = self.report(phase, store)
                self.assertEqual(status, expected); self.assertTrue(manifest['failure_inspection_incomplete'])
                self.assertIn('Failure inspection incomplete: True', decision)

    def test_interruption_has_no_lock_and_owned_artifacts_cannot_be_reopened(self):
        from struct2circuit.calibration.storage import create_validation_lock
        from struct2circuit.calibration.config import SCOPE
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); out = root/'out'
            with OutputStore(out) as store:
                phase = self.phase(store, 'combined'); self.report(phase, store)
                with self.assertRaises(ValueError):
                    create_validation_lock(root/'lock.json', policy=17, scope=SCOPE, reserves=False,
                        budget=phase.budget, development_status='ACCEPTED', development_output=out)
            self.assertFalse((root/'lock.json').exists()); self.assertFalse(list(root.glob('lock.json.*.tmp')))
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            with self.assertRaises(FileExistsError): OutputStore(out)
            self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})

    def interrupted_checks(self, phase, store, *, policy=17, before_first=False):
        from struct2circuit.calibration.decisions import Check
        produced = []
        def construct(name, status, essential=True):
            if before_first or name == 'resource_completion':
                raise MemoryError('synthetic interrupted check construction')
            check = Check(name, status, essential)
            produced.append(check)
            return check
        with patch('struct2circuit.calibration.driver.Check', side_effect=construct):
            status, manifest, decision = self.report(phase, store, policy=policy)
        self.assertTrue(manifest['failure_inspection_incomplete'])
        self.assertIn('Failure inspection incomplete: True', decision)
        self.assertEqual(manifest['selected_policy'], policy)
        return status, manifest, produced

    def test_interrupted_real_E_K_and_combined_checks_preserve_all_nine_failures(self):
        from struct2circuit.calibration.config import SCOPE
        from struct2circuit.calibration.storage import create_validation_lock
        for policy in (9, 17, 33):
            for kind in ('E', 'K', 'combined'):
                with self.subTest(policy=policy, kind=kind), tempfile.TemporaryDirectory() as t:
                    out = Path(t)/'out'
                    with OutputStore(out) as store:
                        phase = self.phase(store, kind, policy)
                        self.assertEqual(phase.status(policy), 'FAILED')
                        if kind == 'K':
                            self.assertTrue(all(c.status == 'pass' for c in phase.checks(policy)[0]
                                                if c.name.endswith(':E')))
                        status, manifest, produced = self.interrupted_checks(phase, store, policy=policy)
                        failed = [c.name for c in produced if c.essential and c.status == 'fail']
                        self.assertTrue(failed)
                        self.assertEqual(status, 'FAILED')
                        self.assertEqual(manifest['failed_checks'], failed)
                        with self.assertRaises(ValueError):
                            create_validation_lock(Path(t)/'lock.json', policy=policy, scope=SCOPE,
                                reserves=False, budget=phase.budget, development_status='ACCEPTED',
                                development_output=out)
                    self.assertFalse((Path(t)/'lock.json').exists())
                    saved = {p.name: p.read_bytes() for p in out.iterdir()}
                    with self.assertRaises(FileExistsError): OutputStore(out)
                    self.assertEqual(saved, {p.name: p.read_bytes() for p in out.iterdir()})

    def test_interruption_before_any_check_preserves_only_existing_latches(self):
        for kind, expected in (('E', 'INCONCLUSIVE'), ('arithmetic', 'FAILED'), ('phase', 'FAILED')):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
                phase = self.phase(store, 'none' if kind == 'phase' else kind)
                if kind == 'phase': phase.phase_failures.append('completed mandatory failure')
                status, manifest, produced = self.interrupted_checks(phase, store, before_first=True)
                self.assertEqual(status, expected)
                self.assertEqual(produced, []); self.assertEqual(manifest['failed_checks'], [])

    def test_interrupted_neutral_unresolved_and_optional_checks_remain_inconclusive(self):
        for kind in ('none', 'unresolved', 'optional'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
                status, manifest, _ = self.interrupted_checks(self.phase(store, kind), store)
                self.assertEqual(status, 'INCONCLUSIVE'); self.assertEqual(manifest['failed_checks'], [])

    def test_interrupted_checks_do_not_latch_other_policy_failures(self):
        for selected in (9, 17, 33):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
                other = next(p for p in (9, 17, 33) if p != selected)
                phase = self.phase(store, 'combined', other)
                self.assertEqual(phase.status(other), 'FAILED')
                status, manifest, _ = self.interrupted_checks(phase, store, policy=selected)
                self.assertEqual(status, 'INCONCLUSIVE'); self.assertEqual(manifest['failed_checks'], [])

    def test_late_failure_after_many_nonfailures_is_seen_during_interruption(self):
        with tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
            phase = self.phase(store, 'E')
            phase.expected_keys = tuple((3000+i, 0, 0) for i in range(100))+phase.expected_keys
            status, manifest, produced = self.interrupted_checks(phase, store)
            self.assertGreater(next(i for i,c in enumerate(produced) if c.status == 'fail'), 64)
            self.assertEqual(status, 'FAILED'); self.assertEqual(manifest['failed_checks'], ['1042:0:0:E'])

    def test_interrupted_diagnostic_cap_does_not_limit_later_checks(self):
        with tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
            phase = self.phase(store, 'K')
            pair = phase.pairs[0]
            phase.pairs = [dict(pair, j=i+1) for i in range(100)]
            status, manifest, produced = self.interrupted_checks(phase, store)
            self.assertEqual(sum(c.status == 'fail' for c in produced), 100)
            self.assertEqual(status, 'FAILED'); self.assertEqual(len(manifest['failed_checks']), 64)
            self.assertTrue(manifest['failed_checks_truncated'])

    def test_interim_pair_failure_can_resolve_without_poisoning_terminal_report(self):
        with tempfile.TemporaryDirectory() as t, OutputStore(Path(t)/'out') as store:
            phase = self.phase(store, 'K')
            first = phase.results[phase.cases[0].canonical_key]
            first.production[17] = replace(first.production[17], value=.1997)
            phase._paired_rows()
            self.assertTrue(any(p['status'] == 'fail' for p in phase.pairs if p['policy'] == 17))
            self.assertEqual(phase.status(17), 'FAILED')
            # Supply completed post-refinement reference evidence, then use the
            # real paired classifier. No scientific evaluation is mocked here.
            first.references['B00'] = replace(first.references['B00'], pooled=.1997,
                route_a=.1997, route_b=.1997,
                candidates=[replace(first.references['B00'].candidates[0], value=.1997)])
            phase._paired_rows()
            self.assertTrue(all(p['status'] == 'pass' for p in phase.pairs if p['policy'] == 17))
            self.assertNotEqual(phase.status(17), 'FAILED')
            status, manifest, _ = self.interrupted_checks(phase, store)
            self.assertEqual(status, 'INCONCLUSIVE'); self.assertEqual(manifest['failed_checks'], [])


class ResourceSerializationTests(unittest.TestCase):
    def test_canonical_stream_exact_bytes_and_hash_for_supported_values(self):
        values = [key_budget().snapshot(), {'unicode':'π\n雪',1:'first','1':'last',
            'array':np.array([1.,float('nan'),float('inf'),-0.]), 'tuple':(np.int64(2),None),
            'set':{2,3},'candidate':safety.WorkflowTests().fixture().key}]
        for value in values:
            expected = canonical_json(value)
            self.assertEqual(''.join(canonical_chunks(value)),expected)
            self.assertEqual(canonical_hash(value),hashlib.sha256(expected.encode()).hexdigest())
            stream=io.StringIO();write_canonical(stream,value)
            self.assertEqual(stream.getvalue(),expected+'\n')

    def test_stream_writes_bounded_batches_and_matches_large_budget(self):
        value = key_budget(6000).snapshot()
        expected = canonical_json(value)+'\n'
        class Sink:
            def __init__(self): self.parts=[]
            def write(self,text): self.parts.append(text)
        sink=Sink();write_canonical(sink,value)
        self.assertEqual(''.join(sink.parts),expected)
        self.assertLess(max(map(len,sink.parts)),66000)
        class NullSink:
            def write(self,text): pass
        tracemalloc.start()
        try:
            write_canonical(NullSink(),value)
            peak=tracemalloc.get_traced_memory()[1]
        finally: tracemalloc.stop()
        self.assertLess(peak, 500000)  # no proportional full JSON/container copy

    def test_restoration_shares_identities_without_mutating_snapshot(self):
        original=key_budget(1000);snapshot=json.loads(canonical_json(original.snapshot()))
        before=canonical_json(snapshot)
        self.assertGreater(len({id(row[0]) for row in snapshot['unique_keys']}),2)
        restored=Budget.from_snapshot(snapshot)
        self.assertEqual(restored.unique,original.unique);self.assertEqual(restored.counts,original.counts)
        self.assertEqual(len({id(k[0]) for k in restored.unique}),2)
        self.assertEqual(canonical_json(snapshot),before)

    def test_streaming_lock_checksum_remains_legacy_compatible(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); b=key_budget(); b.mandatory_complete=True
            gate=safety.LockTests().create(root/'lock.json',budget=b)
            document=json.loads(gate.bytes)
            self.assertEqual(document['sha256'],hashlib.sha256(canonical_json(document['payload']).encode()).hexdigest())
            self.assertEqual(gate.bytes, (canonical_json(document)+'\n').encode())
            # Equal identity values in the payload now share pointers, not values.
            self.assertEqual(len({id(row[0]) for row in gate.payload['budget']['unique_keys']}),2)
            gate.authorize(17)
            gate.payload['budget']['unique_keys'][0][1]='ff'*16
            with self.assertRaisesRegex(ValueError,'altered'):gate.authorize(17)

    def test_gate_file_tampering_without_reparsing_on_authorize(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'lock.json';gate=safety.LockTests().create(path)
            with patch('struct2circuit.calibration.storage.json.loads',side_effect=AssertionError('no reparse')):
                gate.authorize(17)
            path.write_bytes(gate.bytes+b' ')
            with self.assertRaisesRegex(ValueError,'altered'):gate.authorize(17)

    def test_partial_stream_write_never_publishes_lock(self):
        from struct2circuit.calibration.storage import create_validation_lock
        from struct2circuit.calibration.config import SCOPE
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);b=key_budget();b.mandatory_complete=True
            evidence=safety.finalized_mock_development(root/'evidence',b)
            def fail(handle,value): handle.write('{');raise OSError('synthetic disk interruption')
            with patch('struct2circuit.calibration.storage.write_canonical',side_effect=fail):
                with self.assertRaises(OSError):create_validation_lock(root/'lock.json',policy=17,scope=SCOPE,
                    reserves=False,budget=b,development_status='ACCEPTED WITH EXPLICIT LIMITATIONS',development_output=evidence)
            self.assertFalse((root/'lock.json').exists());self.assertFalse(list(root.glob('lock.json.*.tmp')))

    def test_resource_or_cpu_stop_after_lock_write_prevents_publication(self):
        from struct2circuit.calibration.storage import create_validation_lock
        from struct2circuit.calibration.config import SCOPE
        for resource_stop in (False,True):
            with self.subTest(resource_stop=resource_stop),tempfile.TemporaryDirectory() as t:
                root=Path(t);clock=[0.];b=key_budget();b.cpu_clock=lambda:clock[0];b.cpu_start=0.
                b.mandatory_complete=True
                evidence=safety.finalized_mock_development(root/'evidence',b)
                if resource_stop:b.resource_guard=ResourceGuard(root, sample=healthy)
                from struct2circuit.calibration.storage import write_canonical as real_write
                def write(handle,value):
                    real_write(handle,value)
                    if resource_stop:b.resource_guard.sample=lambda p:(32*GIB,30*GIB,24*GIB,10*GIB)
                    else:clock[0]=b.limits.cpu_seconds
                with patch('struct2circuit.calibration.storage.write_canonical',side_effect=write):
                    with self.assertRaises(BudgetExhausted):create_validation_lock(root/'lock.json',policy=17,scope=SCOPE,
                        reserves=False,budget=b,development_status='ACCEPTED WITH EXPLICIT LIMITATIONS',development_output=evidence)
                self.assertFalse((root/'lock.json').exists());self.assertFalse(list(root.glob('lock.json.*.tmp')))


class ResourceGuardTests(unittest.TestCase):
    def test_sample_uses_cgroup_limit_and_usage_not_host_ram(self):
        files={'/proc/meminfo':'MemTotal: 4194304 kB\nMemAvailable: 3145728 kB\n',
               '/proc/self/statm':'100 32', '/sys/fs/cgroup/memory.max':str(GIB),
               '/sys/fs/cgroup/memory.current':str(512*MIB)}
        with patch.object(Path,'read_text',new=lambda p:files[str(p)]), \
             patch.object(Path,'is_file',new=lambda p:str(p) in files), \
             patch('struct2circuit.calibration.resources.shutil.disk_usage',return_value=type('Disk',(),{'free':2*GIB})()), \
             patch('struct2circuit.calibration.resources.os.sysconf',return_value=4096):
            self.assertEqual(resource_sample(['/tmp']), (GIB,512*MIB,32*4096,2*GIB))

    def test_admission_and_throttled_monitoring(self):
        samples=[];clock=[0.]
        def sample(paths):samples.append(paths);return healthy(paths)
        b=key_budget();g=ResourceGuard('/tmp/fresh',sample=sample,clock=lambda:clock[0])
        b.resource_guard=g
        for _ in range(100):b.check_time()
        self.assertEqual(len(samples),1)
        b.counts['objective_requests']+=256;b.check_time();self.assertEqual(len(samples),2)
        clock[0]=1.;b.check_time();self.assertEqual(len(samples),3)
        g.check(b,force=True);self.assertEqual(len(samples),4)

    def test_memory_and_disk_insufficiency_fail_closed(self):
        for sample,reason in [((32*GIB,30*GIB,24*GIB,10*GIB),'resource_memory_headroom'),
                             ((32*GIB,100*MIB,64*MIB,10*GIB),'resource_memory_headroom'),
                             ((32*GIB,30*GIB,64*MIB,100*MIB),'resource_disk_headroom'),
                             ((GIB,GIB,512*MIB,10*GIB),'resource_memory_headroom')]:
            with self.subTest(reason=reason,sample=sample):
                b=key_budget();b.resource_guard=ResourceGuard('/tmp/out',sample=lambda p:sample)
                counts=dict(b.counts)
                with self.assertRaisesRegex(ResourceExhausted,'INCONCLUSIVE'):b.check_time()
                self.assertEqual(b.stop_reason,reason);self.assertEqual(dict(b.counts),counts)
                with self.assertRaises(BudgetExhausted):b.check_time()

    def test_unique_key_finalization_reserve_and_lock_loading_headroom(self):
        b=key_budget(1000)
        # Enough for base reserve, but insufficient after exact-key allowance.
        available=512*MIB+500000
        g=ResourceGuard('/tmp/out',sample=lambda p:(32*GIB,available,64*MIB,10*GIB))
        g.check(force=True)
        with self.assertRaises(ResourceExhausted):g.check(b,force=True)
        with self.assertRaises(ResourceExhausted):g.check(force=True,extra_memory=1000000)

    def test_unknown_information_refuses_before_output_or_simulation(self):
        case=safety.WorkflowTests().fixture()
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'out'
            with patch('struct2circuit.calibration.driver.ResourceGuard',
                       side_effect=lambda *a:ResourceGuard(*a,sample=lambda p:(_ for _ in ()).throw(OSError('missing cgroup')))), \
                 patch('struct2circuit.calibration.driver.PhaseDriver.run',side_effect=AssertionError('no simulation')):
                with self.assertRaisesRegex(ResourceExhausted,'unavailable'):run_synthetic([case],out,limits=Limits())
            self.assertFalse(out.exists())

    def test_resource_stop_during_numerical_work_reports_inconclusive(self):
        case=safety.WorkflowTests().fixture();calls=[0]
        def sample(paths):
            calls[0]+=1
            # Admit, interrupt active work, then permit reporting after release.
            return (32*GIB,30*GIB,24*GIB,10*GIB) if calls[0]==2 else healthy(paths)
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'out'
            with patch('struct2circuit.calibration.driver.ResourceGuard',
                       side_effect=lambda *a:ResourceGuard(*a,sample=sample)):
                status,snapshot=run_synthetic([case],out,limits=Limits(),policies=(9,))
            self.assertEqual(status,'INCONCLUSIVE');self.assertFalse(snapshot['mandatory_complete'])
            self.assertEqual(snapshot['stop_reason'],'resource_memory_headroom')
            self.assertLess(snapshot['counters']['actual_evaluations'],1000)
            self.assertIn('INCONCLUSIVE',(out/'decision.md').read_text())

    def test_persistent_resource_stop_leaves_small_truthful_incomplete_report(self):
        case=safety.WorkflowTests().fixture();calls=[0]
        def sample(paths):
            calls[0]+=1
            return healthy(paths) if calls[0]==1 else (32*GIB,30*GIB,24*GIB,10*GIB)
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'out'
            with patch('struct2circuit.calibration.driver.ResourceGuard',
                       side_effect=lambda *a:ResourceGuard(*a,sample=sample)):
                with self.assertRaises(ResourceExhausted):run_synthetic([case],out,limits=Limits(),policies=(9,))
            manifest=json.loads((out/'calibration_manifest.json').read_text())
            self.assertEqual(manifest['status'],'INCONCLUSIVE');self.assertFalse(manifest['reporting_complete'])
            self.assertFalse(manifest['development_finalized'])
            self.assertEqual((out/'budget.json').stat().st_size,0)
