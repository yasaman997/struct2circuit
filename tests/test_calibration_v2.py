"""Synthetic v2 contract checks. No approved QUBO generator is invoked."""
from dataclasses import asdict, replace
from pathlib import Path
from struct import pack
import json
import os
import subprocess
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
    seal_process, settle_process, settle_publication, current_snapshot, verify_evidence, create_lock,
    runtime_identity, require_official_runtime)
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


class ResourceChainRepairTests(unittest.TestCase):
    def receipt(self,root,name,budget,previous=None,cpu=1.):
        p=root/name;p.mkdir()
        (p/'budget.json').write_text(canonical_json(budget.snapshot()))
        r=finish_resources(p,budget,budget.phase,previous=previous)
        receipt=seal_process(r,process_cpu_seconds=cpu,prior_cpu=budget.prior_cpu)
        # Before the repair the raw cutoff is exposed directly. After the repair
        # the OS-observed phase-exit settlement is required before exposure.
        from struct2circuit.calibration_v2 import storage
        if hasattr(storage,'settle_process'):
            storage.settle_process(receipt,process_cpu_seconds=cpu)
        return p,receipt

    def test_self_consistent_reset_validation_ledger_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dev=Budget();dev.mandatory_complete=True
            dev.request('toy','B00');dev.charge(('model',pack('>dd',.2,.3)),'toy','B00')
            _,prior=self.receipt(root,'development',dev)
            val=Budget(phase='validation');val.prior_cpu=1.;val.prior_wall=dev.snapshot()['wall_seconds'];val.mandatory_complete=True
            with self.assertRaises(ValueError):
                core,receipt=self.receipt(root,'validation',val,previous=prior)
                current_snapshot(core,receipt)

    def test_domain_request_counter_reset_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dev=Budget();dev.mandatory_complete=True
            dev.request('toy','B00');dev.charge(('model',pack('>dd',.2,.3)),'toy','B00')
            _,prior=self.receipt(root,'development',dev)
            s=dev.snapshot();s['cpu_seconds']=1.
            val=Budget.from_snapshot(s,phase='validation');val.mandatory_complete=True
            val.domains['toy','B00','requests']=0
            with self.assertRaises(ValueError):
                core,receipt=self.receipt(root,'validation',val,previous=prior)
                current_snapshot(core,receipt)

    def test_phase_reassignment_rejected_even_with_same_total(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dev=Budget();dev.mandatory_complete=True
            dev.request('toy','B00');dev.charge(('model',pack('>dd',.2,.3)),'toy','B00')
            _,prior=self.receipt(root,'development',dev)
            s=dev.snapshot();s['cpu_seconds']=1.
            val=Budget.from_snapshot(s,phase='validation');val.mandatory_complete=True
            val.phases=dict(development=0,validation=1,diagnostics=0)
            with self.assertRaises(ValueError):
                core,receipt=self.receipt(root,'validation',val,previous=prior)
                current_snapshot(core,receipt)

    def test_landscape_and_actual_domain_reassignment_rejected(self):
        for field in ('landscape','domain'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);dev=Budget();dev.mandatory_complete=True
                dev.request('toy','B00');dev.charge(('model',pack('>dd',.2,.3)),'toy','B00')
                _,prior=self.receipt(root,'development',dev)
                s=dev.snapshot();s['cpu_seconds']=1.
                val=Budget.from_snapshot(s,phase='validation');val.mandatory_complete=True
                if field=='landscape':val.landscapes.clear();val.landscapes['other']=1
                else:
                    val.domains['toy','B00','actual']=0;val.domains['toy','B10','actual']=1
                with self.assertRaises(ValueError):
                    core,receipt=self.receipt(root,'validation',val,previous=prior)
                    current_snapshot(core,receipt)

    def test_valid_three_phase_chain_and_diagnostics_reset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dev=Budget();dev.mandatory_complete=True
            dev.request('toy','B00');dev.charge(('model',pack('>dd',.2,.3)),'toy','B00')
            devcore,devreceipt=self.receipt(root,'development',dev)
            val=Budget.from_snapshot(current_snapshot(devcore,devreceipt),phase='validation');val.mandatory_complete=True
            val.request('validation-toy','B00');val.charge(('val-model',pack('>dd',.2,.3)),'validation-toy','B00')
            valcore,valreceipt=self.receipt(root,'validation',val,previous=devreceipt)
            diag=Budget.from_snapshot(current_snapshot(valcore,valreceipt),phase='diagnostics');diag.mandatory_complete=True
            diag.request('toy','B10');diag.charge(('model',pack('>dd',.4,.3)),'toy','B10',optional=True)
            core,receipt=self.receipt(root,'diagnostics',diag,previous=valreceipt)
            snap=current_snapshot(core,receipt)
            self.assertEqual(snap['phases'],dict(development=1,validation=1,diagnostics=1))
            self.assertEqual(snap['counters']['actual_evaluations'],3)
            bad=Budget(phase='diagnostics');bad.prior_cpu=snap['cpu_seconds'];bad.prior_wall=snap['wall_seconds'];bad.mandatory_complete=True
            with self.assertRaises(ValueError):self.receipt(root,'reset-diagnostics',bad,previous=valreceipt)

    def test_later_phase_cannot_discard_prior_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):self.receipt(Path(temporary),'validation',Budget(phase='validation'))

    def test_unsettled_phase_cannot_expose_grant(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'core';p.mkdir();b=Budget();b.mandatory_complete=True
            (p/'budget.json').write_text(canonical_json(b.snapshot()))
            r=finish_resources(p,b,'development');receipt=seal_process(r,process_cpu_seconds=1.)
            with self.assertRaises((ValueError,FileNotFoundError)):current_snapshot(p,receipt)
            settle_process(receipt,process_cpu_seconds=2.)
            self.assertEqual(current_snapshot(p,receipt)['cpu_seconds'],2.)

    def test_real_process_exit_charges_post_seal_finalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            core=Path(temporary)/'core'
            program='''
import sys
from pathlib import Path
from time import process_time
from struct2circuit.calibration_v2.budget import Budget
from struct2circuit.calibration_v2.storage import canonical_json,finish_resources,seal_process
p=Path(sys.argv[1]);p.mkdir();b=Budget(cpu_origin=0.);b.mandatory_complete=True
(p/'budget.json').write_text(canonical_json(b.snapshot()))
r=finish_resources(p,b,'development')
seal_process(r,process_cpu_seconds=0.,supervisor_cpu_seconds=process_time())
# Simulate real finalization CPU after the old cutoff and receipt fsync.
start=process_time()
while process_time()-start<.1: pass
'''
            env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')
            child=subprocess.Popen([sys.executable,'-c',program,str(core)],env=env)
            _,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
            self.assertEqual(child.returncode,0)
            receipt=Path(str(core)+'.resources.json.process-ended.json')
            from struct2circuit.calibration_v2.storage import _record
            early=_record(receipt)['cpu_seconds'];full=usage.ru_utime+usage.ru_stime
            self.assertGreater(full-early,.08)
            settle_process(receipt,process_cpu_seconds=full)
            self.assertEqual(current_snapshot(core,receipt)['cpu_seconds'],full)

    def test_receipt_sealing_crosses_cpu_ceiling_before_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dev=Budget();dev.mandatory_complete=True
            _,prior=self.receipt(root,'development',dev,cpu=28798.2)
            val=Budget(phase='validation');val.prior_cpu=28798.2;val.prior_wall=dev.snapshot()['wall_seconds'];val.mandatory_complete=True
            p=root/'validation';p.mkdir();(p/'budget.json').write_text(canonical_json(val.snapshot()))
            cutoff=finish_resources(p,val,'validation',previous=prior)
            from struct2circuit.calibration_v2 import storage
            clock=[.1];original=storage.atomic_json
            def expensive_seal(*args,**kwargs):
                result=original(*args,**kwargs);clock[0]+=2.;return result
            with patch.object(storage,'atomic_json',side_effect=expensive_seal):
                receipt=seal_process(cutoff,process_cpu_seconds=1.,supervisor_cpu_seconds=clock[0],prior_cpu=28798.2)
            if hasattr(storage,'settle_process'):
                storage.settle_process(receipt,process_cpu_seconds=1.+clock[0])
            with self.assertRaises(BudgetExhausted):
                Budget.from_snapshot(current_snapshot(p,receipt),phase='diagnostics')


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
            with self.assertRaises((ValueError,FileNotFoundError)):current_snapshot(p,sealed)
            settle_process(sealed,process_cpu_seconds=1.)
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
            cls.driver=d
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
        settle_process(cls.receipt,process_cpu_seconds=10.)

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

    def test_real_negative_evidence_followed_by_landscape_preserves_failed(self):
        result=self.driver.results[1042,0,0]
        original=result.production[33];proofs=self.driver.counterexamples.copy()
        # Fault-inject an actually evaluated poor production choice. The driver
        # emits the genuine negative-evidence record from real n=4 values; this
        # reporting test makes no claim that the real P33 optimizer selects it.
        bad=result.initial_metrics['normalized_gap']
        self.assertGreater(bad-result.references['B00'].pooled,DEFICIT+1e-5)
        try:
            result.production[33]=replace(original,u=0.,beta=0.,value=bad)
            self.driver._capture_failures()
            rows=[self.driver.row(c) for c in self.driver.cases]
            self.assertTrue(rows[0]['negative_evidence'])
            self.assertEqual(assess(rows,self.keys)[0],'FAILED')
        finally:
            result.production[33]=original;self.driver.counterexamples=proofs

    def test_missing_control_objective_identity_rejected(self):
        self.alter_row(lambda r:[c.pop('objective_identity',None) for c in r['controls']])
        with self.assertRaises(ValueError):verify_evidence(self.core)

    def test_cross_landscape_control_substitution_rejected(self):
        # Keep valid values and points; provenance alone must reject substitution.
        self.alter_row(lambda r:[c.update(objective_identity='other-landscape-objective') for c in r['controls']])
        with self.assertRaises(ValueError):verify_evidence(self.core)

    def test_audit_fine_starts_missing_rejected(self):
        def missing(r):
            ref=r['references']['B00']
            ref['attempts']=[a for a in ref['attempts'] if a['source']!='fine_lbfgsb']
        self.alter_row(missing)
        with self.assertRaises(ValueError):verify_evidence(self.core)

    def test_audit_start_substitution_rejected(self):
        def bad(r):
            next(a for a in r['references']['B00']['attempts'] if a['source']=='fine_lbfgsb')['start']=[.123,.456]
        self.alter_row(bad)
        with self.assertRaises(ValueError):verify_evidence(self.core)

    def full_bundle_core(self):
        p=Path(self.temporary.name)/(self._testMethodName+'-core');sim=toy()
        cases=[Case(1042,0,g,0,0,sim,64,audit=g==0) for g in range(2)];b=Budget()
        with OutputStore(p) as store:
            from struct2circuit.calibration.objective import array_hash
            store.row('instances.jsonl',asdict(self.spec)|dict(status='generated',Q=sim.problem.Q,c=sim.problem.c,
                coefficients_sha256=array_hash(sim.problem.Q,sim.problem.c),generator=v1.FAMILIES[0][0],parameters=v1.FAMILIES[0][1]))
            d=PhaseDriver(cases,b,store,expected_keys=self.keys,synthetic=True).run()
            result=d.results[1042,0,0]
            if not result.references['B00'].full_bundle:
                result.search.complete_bundle(result.references['B00'],v1.BY_NAME['B00'],production=result.production[33])
                d._selected_oracles(result)
            self.assertEqual(d.write_results(),'QUALIFIED')
            store.json('calibration_manifest.json',manifest()|dict(phase='development',status='QUALIFIED',phase_status='QUALIFIED',
                reporting_complete=True,provenance={'driver':identity()},runtime=runtime_identity(),development_finalized=True))
            store.sync()
        f=p/'landscapes.jsonl';rows=[json.loads(x) for x in f.read_text().splitlines()]
        for row in rows:row['synthetic']=False
        f.write_text(''.join(canonical_json(r)+'\n' for r in rows))
        return p

    def test_actual_full_bundle_verifies_and_missing_powell_rejected(self):
        p=self.full_bundle_core();self.assertEqual(len(verify_evidence(p)),8)
        f=p/'landscapes.jsonl';rows=[json.loads(x) for x in f.read_text().splitlines()]
        for row in rows:
            ref=row['references']['B00'];self.assertEqual(len([a for a in ref['attempts'] if a['source']=='powell']),2)
            ref['attempts']=[a for a in ref['attempts'] if a['source']!='powell']
        f.write_text(''.join(canonical_json(r)+'\n' for r in rows))
        with self.assertRaises(ValueError):verify_evidence(p)

    def test_full_bundle_fine_starts_and_powell_start_tampering_rejected(self):
        p=self.full_bundle_core();f=p/'landscapes.jsonl';original=f.read_text()
        for source in ('fine_lbfgsb','powell'):
            with self.subTest(source=source):
                rows=[json.loads(x) for x in original.splitlines()]
                for row in rows:
                    a=next(a for a in row['references']['B00']['attempts'] if a['source']==source)
                    a['start']=[.123,.456]
                f.write_text(''.join(canonical_json(r)+'\n' for r in rows))
                with self.assertRaises(ValueError):verify_evidence(p)

    def test_routine_local_attempts_missing_rejected(self):
        def bad(r):
            a=r['references']['B00']['attempts']
            first=next(x for x in a if x['source']=='lbfgsb')
            r['references']['B00']['attempts']=[x for x in a if x['source']!='lbfgsb' or x is first]
        self.alter_row(bad)
        with self.assertRaises(ValueError):verify_evidence(self.core)

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
        settle_publication(lock,process_cpu_seconds=10.)
        gate=ValidationGate(lock)
        self.assertEqual(gate.current_budget()['counters']['actual_evaluations'],json.loads((self.core/'budget.json').read_text())['counters']['actual_evaluations'])
        gate.begin_generation(33)
        with self.assertRaises(ValueError):gate.begin_generation(33)

    def test_coordinator_lock_requires_settlement_covering_publication(self):
        p=Path(self.temporary.name)/self._testMethodName;p.mkdir()
        for name,data in self.original.items():(p/name).write_bytes(data)
        b=Budget.from_snapshot(json.loads((p/'budget.json').read_text()),phase='development')
        r=finish_resources(p,b,'development');receipt=seal_process(r,process_cpu_seconds=5.)
        lock=Path(str(p)+'.lock')
        create_lock(lock,p,receipt,approval='UNIT_TEST_ONLY',_provisional=True)
        with self.assertRaises((ValueError,FileNotFoundError)):ValidationGate(lock)
        settle_process(receipt,process_cpu_seconds=10.,publication=lock)
        gate=ValidationGate(lock)
        self.assertEqual(gate.current_budget()['cpu_seconds'],10.)
        self.assertEqual(gate.grant_path,receipt)

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
