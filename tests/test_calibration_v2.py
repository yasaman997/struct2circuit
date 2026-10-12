"""Synthetic v2 contract checks. No approved QUBO generator is invoked."""
from dataclasses import asdict, replace
from pathlib import Path
from struct import pack
import json
import tempfile
import unittest
from unittest.mock import patch
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from struct2circuit.calibration import config as v1
from struct2circuit.calibration.budget import Budget as V1Budget, Limits as V1Limits, BudgetExhausted
from struct2circuit.calibration.decisions import Check, band, DEFICIT, DISTORTION, reference_status
from struct2circuit.calibration.driver import Case
from struct2circuit.calibration.objective import Objective
from struct2circuit.calibration.reference import Candidate, ReferenceResult, ReferenceSearch
from struct2circuit.calibration.storage import OutputStore, canonical_hash, canonical_json
from struct2circuit.calibration_v2.budget import Budget, Limits
from struct2circuit.calibration_v2.config import PROTOCOL_HASH, expected_keys, expected_pairs, manifest, identity
from struct2circuit.calibration_v2.driver import PhaseDriver, generate_cases
from struct2circuit.calibration_v2.evidence import classify, reference_values, assess, required_work
from struct2circuit.calibration_v2.storage import (ValidationGate, atomic_json, authorize, finish_resources,
    seal_process, current_snapshot, verify_evidence, create_lock, runtime_identity, require_official_runtime)
from struct2circuit.calibration_v2.diagnostics import contradiction_status, addendum, DiagnosticDriver
from struct2circuit.mixers import ring_mixer
from struct2circuit.problems import CardinalityQUBO
from struct2circuit.simulator import FeasibleSubspaceQAOA


def toy(init='uniform',constant=False):
    q=np.diag([.1,.7,-.5,1.]);q[0,2]=q[2,0]=.23
    c=np.array([-.4,.1,.3,-.2])
    if constant:q,c=np.zeros((4,4)),np.ones(4)
    return FeasibleSubspaceQAOA(CardinalityQUBO(q,c,2),ring_mixer(4),initialization=init)


class MetadataTests(unittest.TestCase):
    def test_full_prospective_matrix(self):
        m=manifest()
        self.assertEqual(m['policy'],33);self.assertEqual(m['history']['ineligible_policies'],[9,17])
        self.assertEqual({i for i,g,s in m['development_keys']},set(range(8)))
        self.assertEqual(len(m['development_keys']),192);self.assertEqual(len(m['development_pairs']),672)
        self.assertEqual(len(m['validation_keys']),96);self.assertEqual(len(m['validation_pairs']),336)
        self.assertEqual(m['settings']['essential_domains'],['B00'])

    def test_preview_does_not_generate(self):
        with patch('struct2circuit.problems.block_correlated_qubo',side_effect=AssertionError('No official generation')):
            manifest()

    def test_versioned_sources_do_not_change_v1_identity(self):
        self.assertEqual(identity()['v1_sources_sha256'],'c1d0419129380837feac5f100617f3e0c00bbbb0cb9f3fb1f37575f472b9cfb2')


class BudgetTests(unittest.TestCase):
    def make(self,**kw):
        return Budget(Limits(development=3,total=5,validation_reserve=2,landscape=5,cpu_seconds=30),**kw)

    def charge(self,b,i,landscape='toy',oracle=False,optional=False):
        b.request(landscape,'B00');b.charge(('model',pack('>dd',i,.2)),landscape,'B00',oracle=oracle,optional=optional)

    def test_development_cannot_borrow(self):
        b=self.make()
        for i in range(3):self.charge(b,i)
        with self.assertRaises(BudgetExhausted):self.charge(b,3)
        self.assertEqual(b.counts['actual_evaluations'],3);self.assertEqual(b.stop_reason,'development_evaluation_ceiling')

    def test_cumulative_validation_no_reset(self):
        b=self.make();self.charge(b,0,oracle=True)
        s=json.loads(canonical_json(b.snapshot()));c=Budget.from_snapshot(s,phase='validation')
        for i in range(1,5):self.charge(c,i)
        with self.assertRaises(BudgetExhausted):self.charge(c,5)
        self.assertEqual(c.phases,dict(development=1,validation=4,diagnostics=0))
        self.assertEqual(c.counts['oracle_calls'],1)

    def test_cache_hits_and_fresh_repeats(self):
        b=self.make();o=Objective(toy(),b,'toy')
        o.evaluate(.2,.3);o.evaluate(.2,.3);o.evaluate(.2,.3,fresh=True)
        self.assertEqual(b.counts['actual_evaluations'],2);self.assertEqual(b.counts['cache_hits'],1)
        self.assertEqual(len(b.unique),1)

    def test_landscape_limit(self):
        b=Budget(Limits(development=3,total=5,validation_reserve=2,landscape=1,cpu_seconds=30))
        self.charge(b,0)
        with self.assertRaises(BudgetExhausted):self.charge(b,1)
        self.assertEqual(b.stop_reason,'landscape_evaluation_ceiling')

    def test_cpu_sticky_and_prior_cost(self):
        clock=[0.];b=self.make(cpu_clock=lambda:clock[0],wall_clock=lambda:clock[0]);b.prior_cpu=29.
        clock[0]=1.
        with self.assertRaises(BudgetExhausted):b.check_time()
        clock[0]=0.
        with self.assertRaises(BudgetExhausted):b.check_time()

    def test_no_v1_partition_in_v2(self):
        b=self.make()
        for i in range(3):
            b.request('toy','B00');b.charge(('toy',bytes([i])),'toy','B00','shared')
        self.assertEqual(b.counts['shared'],3)
        v=V1Budget(V1Limits(essential=1,shared=1,total=2,landscape=2,optional=1,cpu_seconds=30))
        for i in range(2):v.request('toy','B00');v.charge(('toy',bytes([i])),'toy','B00')
        self.assertEqual(v.counts['essential'],1);self.assertEqual(v.counts['shared'],1)

    def test_reject_v1_snapshot_and_corrupt_accounting(self):
        with self.assertRaises(ValueError):Budget.from_snapshot(V1Budget().snapshot(),phase='validation')
        for path,value in [('cpu_seconds',float('nan')),('remaining',100)]:
            s=self.make().snapshot();s[path]=value
            with self.assertRaises(ValueError):Budget.from_snapshot(s,phase='validation')
        s=self.make().snapshot();s['phases']['development']=1
        with self.assertRaises(ValueError):Budget.from_snapshot(s,phase='validation')

    def test_limits_protect_allocation_and_cannot_increase(self):
        for kw in (dict(total=8_000_001),dict(development=4_500_001),dict(landscape=120_001),dict(cpu_seconds=28_801),dict(development=True),dict(total=7_000_000)):
            with self.assertRaises(ValueError):Limits(**kw)

    def test_optional_not_allowed_before_validation(self):
        with self.assertRaises(ValueError):self.charge(self.make(),0,optional=True)
        s=self.make().snapshot()
        with self.assertRaises(ValueError):Budget.from_snapshot(s,phase='diagnostics')


class DecisionTests(unittest.TestCase):
    def test_exact_guard_endpoints(self):
        for threshold in (DEFICIT,DISTORTION):
            lo,hi=threshold-1e-5,threshold+1e-5
            self.assertEqual(band(lo,threshold),'ambiguous');self.assertEqual(band(hi,threshold),'ambiguous')
            self.assertEqual(band(np.nextafter(lo,-np.inf),threshold),'pass')
            self.assertEqual(band(np.nextafter(hi,np.inf),threshold),'fail')
        self.assertEqual(reference_status(1e-5),'pass')
        self.assertEqual(reference_status(np.nextafter(1e-5,np.inf)),'unresolved')

    def test_failure_precedence(self):
        self.assertEqual(classify([Check('E','fail')],['E','missing']),'FAILED')
        self.assertEqual(classify([Check('E','pass')],['E','missing']),'INCONCLUSIVE')
        self.assertEqual(classify([Check('E','pass')],['E']),'QUALIFIED')

    def test_reference_route_independence(self):
        ref=ReferenceResult('B00',candidates=[Candidate(.2,.3,.1,'A','local'),Candidate(.3,.3,.2,'B','de'),
            Candidate(.4,.3,.1,'pool','injected')],route_a=.1,route_b=.2,pooled=.1,r=.1,faces_complete=True,complete=True)
        self.assertFalse(reference_values(ref.record())[1])
        ref.candidates[1]=replace(ref.candidates[1],value=.1);ref.route_b=.1;ref.r=0.
        self.assertTrue(reference_values(ref.record())[1])
        ref.refined=True;ref.coarse=.11;ref.f=.01
        self.assertFalse(reference_values(ref.record())[1])

    def test_complete_flag_cannot_override_missing_routes(self):
        ref=ReferenceResult('B00',complete=True,faces_complete=True,pooled=.1)
        self.assertFalse(reference_values(ref.record())[1])

    def test_broader_incompleteness_separate(self):
        self.assertEqual(contradiction_status('QUALIFIED',inside=False,reliable=True,recalculated='FAILED'),'QUALIFIED')
        self.assertEqual(contradiction_status('QUALIFIED',inside=True,reliable=True,recalculated='FAILED'),'FAILED')
        self.assertEqual(contradiction_status('QUALIFIED',unchecked=True),'INCONCLUSIVE')
        self.assertEqual(contradiction_status('FAILED',unchecked=True),'FAILED')
        self.assertEqual(contradiction_status('QUALIFIED',shared_failure=True),'FAILED')


class SyntheticWorkflowTests(unittest.TestCase):
    def test_actual_b00_workflow_and_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            sim=toy();cases=[Case(1042,0,g,0,0,sim,64,audit=g==0) for g in range(2)]
            b=Budget(Limits(development=30000,total=60000,validation_reserve=30000,landscape=30000,cpu_seconds=60))
            with OutputStore(Path(temporary)/'out') as store:
                d=PhaseDriver(cases,b,store,expected_keys=[c.key for c in cases],synthetic=True).run()
                self.assertEqual(set(d.results[1042,0,0].references),{'B00'})
                self.assertEqual(set(d.results[1042,0,0].production),{33})
                self.assertEqual(len(d.pairs),1)
                self.assertEqual(d.status(),'QUALIFIED')
                self.assertTrue(d.results[1042,0,0].references['B00'].refined)
                self.assertTrue(d.results[1042,0,0].controls)
                with self.assertRaises(ValueError):d.add_policy(17)
                self.assertEqual(d.write_results(),'QUALIFIED')

    def test_resource_exhaustion_truthful(self):
        with tempfile.TemporaryDirectory() as temporary:
            case=Case(1042,0,0,0,0,toy(),64)
            b=Budget(Limits(development=1,total=2,validation_reserve=1,landscape=1,cpu_seconds=30))
            with OutputStore(Path(temporary)/'out') as store:
                d=PhaseDriver([case],b,store,expected_keys=[case.key],synthetic=True).run()
                self.assertEqual(d.status(),'INCONCLUSIVE')
                self.assertEqual(b.counts['actual_evaluations'],1)

    def test_constant_not_optimized(self):
        with tempfile.TemporaryDirectory() as temporary:
            case=Case(1042,0,0,0,0,toy(constant=True),64)
            with OutputStore(Path(temporary)/'out') as store:
                d=PhaseDriver([case],Budget(),store,expected_keys=[case.key],synthetic=True).run()
                self.assertEqual(d.status(),'INCONCLUSIVE');self.assertEqual(d.budget.counts['actual_evaluations'],0)

    def test_actual_oracle_failure_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            case=Case(1042,0,0,0,0,toy(),64)
            with OutputStore(Path(temporary)/'out') as store:
                d=PhaseDriver([case],Budget(),store,expected_keys=[case.key],synthetic=True)
                with patch('struct2circuit.calibration.diagnostics.expm',side_effect=lambda x:np.eye(len(x))):d.run()
                self.assertEqual(d.status(),'FAILED')

    def test_completed_deficit_survives_unfinished_refinement(self):
        with tempfile.TemporaryDirectory() as temporary:
            case=Case(1042,0,0,0,0,toy(),64)
            with OutputStore(Path(temporary)/'out') as store:
                d=PhaseDriver([case],Budget(),store,expected_keys=[case.key],synthetic=True).run()
                r=d.results[case.key]
                r.production[33]=replace(r.production[33],value=r.references['B00'].pooled+.001)
                d._capture_failures()
                r.references['B00'].complete=False;r.references['B00'].faces_complete=False
                r.references['B00'].r=None;r.error='interrupted_refinement'
                d.budget.stop_reason='landscape_evaluation_ceiling'
                self.assertEqual(d.status(),'FAILED')


class StorageTests(unittest.TestCase):
    def test_official_thread_strings_required(self):
        with patch.dict('os.environ',{'OPENBLAS_NUM_THREADS':'2'}):
            with self.assertRaises(ValueError):require_official_runtime()

    def test_actual_loaded_pool_rejected(self):
        env={n:'1' for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}
        with patch.dict('os.environ',env),patch('struct2circuit.calibration_v2.storage.threadpool_info',return_value=[{'num_threads':2}]):
            with self.assertRaisesRegex(ValueError,'Loaded numerical'):require_official_runtime()

    def test_official_frozen_versions_required(self):
        env={n:'1' for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}
        with patch.dict('os.environ',env),patch('struct2circuit.calibration_v2.storage.threadpool_info',return_value=[{'num_threads':1}]),patch('struct2circuit.calibration_v2.storage.identity',return_value=dict(python='3.11.0',numpy='2.5.3',scipy='1.18.1')):
            with self.assertRaisesRegex(ValueError,'protocol-frozen'):require_official_runtime()

    def test_invalid_authorization_before_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'auth.json';path.write_text('{}')
            with patch('struct2circuit.problems.block_correlated_qubo',side_effect=AssertionError('No approved QUBO')):
                with self.assertRaises(PermissionError):generate_cases('development',Budget(),authorization=path,output=Path(temporary)/'out')
            self.assertFalse((Path(temporary)/'out').exists())

    def test_v1_lock_rejected_even_with_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'lock';v=dict(schema_version=1,policy=17)
            atomic_json(p,dict(payload=v,sha256=canonical_hash(v)))
            with self.assertRaises(ValueError):ValidationGate(p)

    def test_corrupted_lock_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'lock';atomic_json(p,dict(payload={'schema_version':2},sha256='wrong'))
            with self.assertRaises(ValueError):ValidationGate(p)

    def test_atomic_publication_and_sentinel(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'x';atomic_json(p,{'a':1});before=p.read_bytes()
            with self.assertRaises(FileExistsError):atomic_json(p,{'a':2})
            self.assertEqual(p.read_bytes(),before)

    def test_one_time_marker_durable_and_shared_by_copied_lock(self):
        # Capability mechanics tested with an unrelated fixture, scientific
        # verifier independently covered below; no official generator is called.
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'lock';p.write_text('fixture')
            gate=object.__new__(ValidationGate);gate.path=p
            from struct2circuit.calibration_v2.storage import file_hash
            gate.bytes_hash=file_hash(p);gate.payload={'policy':33,'driver':identity(),'development_evidence':{'directory':'toy','sha256':{}},'resource_path':'toy'}
            gate.payload_hash=canonical_hash(gate.payload);gate.resource_path=Path(temporary)/'receipt'
            gate.generation_claimed=False;gate.started=False
            with patch('struct2circuit.calibration_v2.storage.verify_evidence'),patch.object(gate,'current_budget',return_value=Budget().snapshot()):
                gate.begin_generation(33)
                marker=Path(str(gate.resource_path)+'.validation-consumed.json');before=marker.read_bytes()
                with self.assertRaises(ValueError):gate.begin_generation(33)
                gate.generation_claimed=False
                with self.assertRaises(FileExistsError):gate.begin_generation(33)
                self.assertEqual(marker.read_bytes(),before)

    def test_addendum_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'original';p.write_bytes(b'original core')
            a=Path(temporary)/'addendum.json'
            addendum(a,cores={'original':'hash'},lock_sha256='hash',evidence=[{'inside_B00':True}],effective_status='FAILED',budget=Budget())
            self.assertEqual(p.read_bytes(),b'original core')
            d=json.loads(a.read_text());self.assertEqual(d['sha256'],canonical_hash(d['payload']))
            self.assertEqual(d['payload']['effective_status'],'FAILED')

    def test_full_process_receipt_and_no_unsealed_grant(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'core';p.mkdir();b=Budget();b.mandatory_complete=True
            (p/'budget.json').write_text(canonical_json(b.snapshot()))
            record=finish_resources(p,b,'development')
            with self.assertRaises(ValueError):current_snapshot(p,record)
            sealed=seal_process(record,process_cpu_seconds=1.)
            self.assertEqual(current_snapshot(p,sealed)['cpu_seconds'],1.)


class EvidenceVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory()
        cls.core=Path(cls.temporary.name)/'core'
        cls.spec=v1.InstanceSpec(1042,0,'development',4,2,123)
        sim=toy();cases=[Case(1042,0,g,0,0,sim,64,audit=g==0) for g in range(2)]
        cls.keys=tuple(c.key for c in cases)
        b=Budget()
        with OutputStore(cls.core) as store:
            from struct2circuit.calibration.objective import array_hash
            store.row('instances.jsonl',asdict(cls.spec)|dict(status='generated',Q=sim.problem.Q,c=sim.problem.c,
                coefficients_sha256=array_hash(sim.problem.Q,sim.problem.c),generator=v1.FAMILIES[0][0],parameters=v1.FAMILIES[0][1]))
            d=PhaseDriver(cases,b,store,expected_keys=cls.keys,synthetic=True).run()
            assert d.write_results()=='QUALIFIED'
            store.json('calibration_manifest.json',manifest()|dict(phase='development',status='QUALIFIED',phase_status='QUALIFIED',
                reporting_complete=True,provenance={'driver':identity()},runtime=runtime_identity(),development_finalized=True))
            store.sync()
        # The verifier is exercised against a substituted small metadata contract,
        # never against official coefficients. This removes the toy-only marker
        # because official evidence rejects that marker; all scalars are real toy
        # evaluations. Its actual n=4 coefficients and identities stay unchanged.
        path=cls.core/'landscapes.jsonl'
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        for row in rows:row['synthetic']=False
        path.write_text(''.join(canonical_json(r)+'\n' for r in rows))
        cls.original={p.name:p.read_bytes() for p in cls.core.iterdir()}
        r=finish_resources(cls.core,b,'development')
        cls.receipt=seal_process(r,process_cpu_seconds=10.)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def setUp(self):
        for name,data in self.original.items():(self.core/name).write_bytes(data)
        self.patches=[patch('struct2circuit.calibration_v2.storage.specs',return_value=(self.spec,)),
                      patch('struct2circuit.calibration_v2.storage.expected_keys',return_value=self.keys),
                      patch('struct2circuit.calibration_v2.storage.require_official_runtime',return_value=runtime_identity()),
                      patch('struct2circuit.calibration.resources.ResourceGuard.check')]
        for p in self.patches:p.start()
        self.addCleanup(lambda:[p.stop() for p in self.patches])

    def alter_row(self,change):
        p=self.core/'landscapes.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()]
        for row in rows:change(row)
        p.write_text(''.join(canonical_json(r)+'\n' for r in rows))

    def test_positive_evidence_reconstructed(self):
        self.assertEqual(len(verify_evidence(self.core)),8)

    def test_missing_control_rejected(self):
        self.alter_row(lambda r:r.update(controls=[c for c in r['controls'] if c['kind']!='gradient']))
        with self.assertRaises(ValueError):verify_evidence(self.core)

    def test_false_qualified_flag_rejected(self):
        self.alter_row(lambda r:r['references']['B00'].update(candidates=[]))
        with self.assertRaises(ValueError):verify_evidence(self.core)

    def test_grid_hole_rejected(self):
        def hole(r):
            c=r['references']['B00']['candidates'];c.pop(next(i for i,v in enumerate(c) if v['source']=='grid'))
        self.alter_row(hole)
        with self.assertRaises(ValueError):verify_evidence(self.core)

    def test_route_seed_substitution_rejected(self):
        def bad(r):
            next(a for a in r['references']['B00']['attempts'] if a['source']=='de_0')['seed']=1
        self.alter_row(bad)
        with self.assertRaises(ValueError):verify_evidence(self.core)

    def test_boundary_evidence_missing_rejected(self):
        self.alter_row(lambda r:r.update(trace_cache=[]))
        with self.assertRaises(ValueError):verify_evidence(self.core)

    def test_pair_missing_or_changed_rejected(self):
        p=self.core/'paired_contrasts.csv';p.write_text(p.read_text().splitlines()[0]+'\n')
        with self.assertRaises(ValueError):verify_evidence(self.core)

    def test_artifact_hash_tampering_rejected(self):
        hashes=verify_evidence(self.core)
        p=self.core/'decision.md';p.write_bytes(p.read_bytes()+b'altered')
        with self.assertRaises(ValueError):verify_evidence(self.core,hashes=hashes)

    def test_complete_lock_and_one_time_capability(self):
        lock=Path(self.temporary.name)/self._testMethodName
        create_lock(lock,self.core,self.receipt,approval='UNIT_TEST_ONLY')
        gate=ValidationGate(lock)
        self.assertEqual(gate.current_budget()['counters']['actual_evaluations'],json.loads((self.core/'budget.json').read_text())['counters']['actual_evaluations'])
        gate.begin_generation(33)
        with self.assertRaises(ValueError):gate.begin_generation(33)

    def test_persisted_models_restore_without_generators(self):
        from struct2circuit.calibration_v2.diagnostics import restore_models
        b=Budget(phase='diagnostics');b.mandatory_complete=True
        with patch('struct2circuit.problems.block_correlated_qubo',side_effect=AssertionError('No official generation')):
            cases,results=restore_models(self.core,b)
        self.assertEqual(len(cases),2);self.assertEqual(len(results),1)
        self.assertTrue(results[1042,0,0].case.audit)
        self.assertEqual(results[1042,0,0].objective.identity,json.loads((self.core/'landscapes.jsonl').read_text().splitlines()[0])['objective_identity'])

    def test_diagnostic_admission_uses_cache_not_key_inventory(self):
        from struct2circuit.calibration_v2.diagnostics import restore_models
        b=Budget(phase='diagnostics');b.mandatory_complete=True
        cases,results=restore_models(self.core,b);r=next(iter(results.values()))
        # Even a complete prior inventory supplies no restored values/states.
        b.unique={r.objective.key(u,beta) for u in v1.grid_axes(v1.BY_NAME['B22'],64)[0]
                  for beta in v1.grid_axes(v1.BY_NAME['B22'],64)[1]}
        b.landscapes[r.objective.landscape]=119999
        driver=DiagnosticDriver(cases,results,b,None)
        self.assertFalse(driver.admit(r,v1.BY_NAME['B22'],0))
        self.assertEqual(driver.not_run[0]['status'],'NOT_RUN')

    def test_reference_and_control_tampering_cannot_issue_lock(self):
        self.alter_row(lambda r:r['references']['B00'].update(route_b=-1.))
        lock=Path(self.temporary.name)/self._testMethodName
        with self.assertRaises(ValueError):create_lock(lock,self.core,self.receipt,approval='UNIT_TEST_ONLY')
        self.assertFalse(lock.exists())


if __name__=='__main__':unittest.main()
