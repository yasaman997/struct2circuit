"""Adversarial scalar fixtures only: no generated QUBOs or QAOA searches."""
from copy import deepcopy
from pathlib import Path
from struct import pack
from types import SimpleNamespace
import json
import subprocess
import tempfile
import unittest
from dataclasses import replace

import numpy as np

from test_alignment_experiment import scalar_arm, toy, unknown_rows
import test_alignment_experiment as base
from struct2circuit.alignment_experiment import config, evidence, selection, storage
from struct2circuit.alignment_experiment.budget import Budget, ArmExhausted
from struct2circuit.alignment_experiment.reference import ReferenceSearch, candidate_id, interior, seal_control
from struct2circuit.alignment_experiment.provenance import read_results, reconcile
from struct2circuit.calibration.config import BY_NAME
from struct2circuit.calibration.storage import canonical_hash
from struct2circuit.calibration_v2.storage import atomic_json
from struct2circuit.mixers import MixerSpec, xy_mixer_hamiltonian
from struct2circuit.problems import CardinalityQUBO


SPEC_FIELDS=('objective_identity','index','route','source','seed','start','start_candidate_id',
             'bounds','settings','objective_call_cap')


def resign_attempt(a):
    a['attempt_id']=canonical_hash({k:a[k] for k in SPEC_FIELDS})


def materialize(row, budget, identity, arm):
    """Write compact synthetic admissions/results for an explicit scalar fixture."""
    cache={};old_to_new={};reference=row['reference']
    coarse_ids=[c['candidate_id'] for c in row['coarse_reference']['candidates']]
    def native(point, fresh=False):
        key=identity,pack('>dd',*map(float,point))
        request=budget.request(arm,'B00',key=key,fresh=fresh)
        if not fresh and key in cache:
            computation=cache[key];budget.resolve(request,computation,cached=True)
        else:
            computation=budget.charge(key,arm,'B00',request_id=request,fresh=fresh)
            budget.outcome(computation,dict(gap=.2,state_sha256='synthetic-state'))
            budget.resolve(request,computation);cache[key]=computation
        return request,computation
    def derivative(point):
        cid=budget.charge((identity,pack('>dd',*map(float,point))),arm,'B00',derivative=True)
        budget.outcome(cid,[0.,0.]);return cid
    def candidate(c, attempt=None):
        old=c['candidate_id'];rid,cid=native((c['u'],c['beta']),c['observation']=='terminal')
        c.update(objective_identity=identity,request_id=rid,computation_id=cid,attempt_id=attempt)
        c['candidate_id']=candidate_id(c);old_to_new[old]=c['candidate_id']
    for c in reference['candidates']:
        if not c['attempt_id']:candidate(c)
    for a in reference['attempts']:
        old=a['attempt_id'];a['objective_identity']=identity
        if a['start_candidate_id']:a['start_candidate_id']=old_to_new[a['start_candidate_id']]
        resign_attempt(a)
        with budget.trace() as work:
            for c in reference['candidates']:
                if c['attempt_id']==old:
                    candidate(c,a['attempt_id'])
                    if c['observation']=='solver' and a['source'] in ('lbfgsb','fine_lbfgsb','de_polish_0','de_polish_1'):
                        derivative((c['u'],c['beta']))
        a['observations']=[old_to_new[c] for c in a['observations']]
        a['terminal_candidate_id']=old_to_new[a['terminal_candidate_id']]
        a.update(work=work,actual_evaluations=len(work['computations']))
    by_id={c['candidate_id']:c for c in reference['candidates']}
    row['coarse_reference']['candidates']=[by_id[old_to_new[cid]] for cid in coarse_ids]
    for i,check in enumerate(row['controls']):
        point=check['point'];kind=check['kind']
        if check.get('candidate_id'):check['candidate_id']=old_to_new[check['candidate_id']]
        with budget.trace() as work:
            if kind=='gradient':
                native(point);derivative(point);native(point)
                for axis,step in enumerate(check['steps']):
                    for sign in (1,-1):
                        p=list(point);p[axis]+=sign*step;native(p)
            elif kind=='direct_expm':
                key=identity,pack('>dd',*map(float,point))
                req=budget.request(arm,'B00',key=key,category='oracle',fresh=True)
                cid=budget.charge(key,arm,'B00',oracle=True,request_id=req,fresh=True)
                budget.counts['matrix_exponentials']+=2
                budget.outcome(cid,dict(gap=.2,state_sha256='synthetic-oracle'),matrix_exponentials=2);budget.resolve(req,cid)
                _,other=native(point)
                check.update(oracle_computation_id=cid,native_computation_id=other,native=dict(normalized_gap=.2))
            elif kind=='candidate_reevaluation':
                _,cid=native(point,True);check['checked_computation_id']=cid
            elif kind=='cyclic_transport':
                _,cid=native(point)
                tid=check.get('transported_identity','synthetic-transport')
                key=tid,pack('>dd',*map(float,point));req=budget.request(arm,'B00',key=key)
                other=budget.charge(key,arm,'B00',request_id=req)
                budget.outcome(other,dict(gap=.2,state_sha256='synthetic-transport'));budget.resolve(req,other)
                check.update(native_computation_id=cid,transported_computation_id=other,transported_identity=tid)
        bound=check.pop('bindings')
        row['controls'][i]=seal_control(SimpleNamespace(budget=budget),check,work)|dict(bindings=bound)
    return row


def journal_fixture(root):
    budget=Budget(root/'evaluations.jsonl',cpu_clock=lambda:0.)
    row=scalar_arm();row['bindings']['objective_identity']='toy-identity'
    materialize(row,budget,'toy-identity','0:0:low')
    records=[dict(family=0,replicate=0,arms=dict(low=row))]
    storage.checkpoint(root,budget,'final');budget.close()
    return records


def load_provenance(root):
    admissions=[json.loads(line)['payload'] for line in (root/'evaluations.jsonl').read_text().splitlines()]
    return read_results(root/'evaluations.results.jsonl',admissions)


class AttemptAndFreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid=scalar_arm()
        source=subprocess.check_output(['git','show','8db4d2d30ad56df3249839455cbd57f5eeca02db:src/struct2circuit/alignment_experiment/evidence.py'],cwd=config.ROOT,text=True)
        namespace={'__package__':'struct2circuit.alignment_experiment'}
        exec(compile(source,'reviewed-evidence.py','exec'),namespace)
        cls.previous=staticmethod(namespace['verify_reference'])

    def unaccepted(self,row):
        try: value,_=evidence.verify_reference(row,0,0)
        except ValueError:return
        self.assertIsNone(value)

    def missing(self,sources):
        row=deepcopy(self.valid)
        row['reference']['candidates']=[c for c in row['reference']['candidates'] if c['source'] not in sources]
        row['coarse_reference']['candidates']=[c for c in row['coarse_reference']['candidates'] if c['source'] not in sources]
        self.assertEqual(self.previous(row,0,0),(.2,False))
        self.unaccepted(row)

    def test_A_missing_local_observations(self):self.missing({'lbfgsb','fine_lbfgsb'})
    def test_B_missing_DE1_and_polish_observations(self):self.missing({'de_1','de_polish_0','de_polish_1'})
    def test_C_missing_Powell_observations(self):self.missing({'powell'})

    def test_D_early_attempt_limit(self):
        row=deepcopy(self.valid)
        for a in row['reference']['attempts']:a.update(interrupted='AttemptLimit',message='solver_objective_call_cap',objective_calls=1)
        self.assertEqual(self.previous(row,0,0),(.2,False));self.unaccepted(row)

    def test_E_Powell_cannot_manufacture_independent_route(self):
        row=deepcopy(self.valid);a=next(a for a in row['reference']['attempts'] if a['source']=='powell')
        old=a['attempt_id'];a['route']='B';resign_attempt(a)
        for c in row['reference']['candidates']:
            if c['attempt_id']==old:
                c.update(route='B',attempt_id=a['attempt_id']);c['candidate_id']=candidate_id(c)
        observed=[c for c in row['reference']['candidates'] if c['attempt_id']==a['attempt_id']]
        a['observations']=[c['candidate_id'] for c in observed if c['observation']=='solver']
        a['terminal_candidate_id']=next(c['candidate_id'] for c in observed if c['observation']=='terminal')
        for check in row['controls'][22:24]:check['point']=list(a['start'])
        self.assertEqual(self.previous(row,0,0),(.2,False));self.unaccepted(row)

    def test_F_unrelated_DE_polish_start(self):
        row=deepcopy(self.valid);a=next(a for a in row['reference']['attempts'] if a['source']=='de_polish_1')
        a['start']=[.4,.5];old=a['attempt_id'];resign_attempt(a)
        for c in row['reference']['candidates']:
            if c['attempt_id']==old:c['attempt_id']=a['attempt_id'];c['candidate_id']=candidate_id(c)
        observed=[c for c in row['reference']['candidates'] if c['attempt_id']==a['attempt_id']]
        a['observations']=[c['candidate_id'] for c in observed if c['observation']=='solver']
        a['terminal_candidate_id']=next(c['candidate_id'] for c in observed if c['observation']=='terminal')
        self.assertEqual(self.previous(row,0,0),(.2,False));self.unaccepted(row)

    def test_G_self_consistent_wrong_fresh_value(self):
        row=deepcopy(self.valid)
        for c in row['controls']:
            if c['kind']=='candidate_reevaluation':c.update(value=.9,checked_value=.9,error=0.)
        self.assertEqual(self.previous(row,0,0),(.2,False));self.unaccepted(row)

    def test_fresh_roles_cannot_be_replaced_by_duplicates(self):
        row=deepcopy(self.valid)
        checks=[c for c in row['controls'] if c['kind']=='candidate_reevaluation']
        checks[1].update(deepcopy(checks[0]));self.unaccepted(row)

    def test_solver_flag_alone_does_not_veto_complete_work(self):
        row=deepcopy(self.valid)
        for a in row['reference']['attempts']:a['success']=False
        for c in row['reference']['candidates']:
            if c['observation']=='terminal':
                old=c['candidate_id'];c['success']=False;c['candidate_id']=candidate_id(c)
                for a in row['reference']['attempts']:
                    if a['terminal_candidate_id']==old:a['terminal_candidate_id']=c['candidate_id']
        self.assertEqual(evidence.verify_reference(row,0,0),(.2,False))


class JournalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory();cls.root=Path(cls.temporary.name)
        cls.records=journal_fixture(cls.root);cls.provenance=load_provenance(cls.root)
    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def test_H_empty_journal_cannot_support_complete_reference(self):
        self.assertEqual(evidence.verify_reference(self.records[0]['arms']['low'],0,0),(.2,False))
        empty={k:[] if k=='records' else {} for k in self.provenance}
        with self.assertRaisesRegex(ValueError,'request'):reconcile(self.records,empty)

    def test_I_other_arm_events_rejected(self):
        bad=deepcopy(self.records);bad[0]['arms']['high']=bad[0]['arms'].pop('low')
        with self.assertRaises(ValueError):reconcile(bad,self.provenance)

    def test_J_wrong_coordinates_and_objective_rejected(self):
        for change in (dict(u=.123),dict(objective_identity='substituted')):
            bad=deepcopy(self.records);bad[0]['arms']['low']['reference']['candidates'][0].update(change)
            if 'objective_identity' in change:bad[0]['arms']['low']['bindings']['objective_identity']='substituted'
            with self.assertRaises(ValueError):reconcile(bad,self.provenance)

    def test_K_missing_oracle_or_derivative_admission(self):
        for category in ('oracle','derivative'):
            bad=deepcopy(self.provenance)
            kind='direct_expm' if category=='oracle' else 'gradient'
            control=next(c for c in self.records[0]['arms']['low']['controls'] if c['kind']==kind)
            cid=next(cid for cid in control['work']['computations'] if bad['computations'][cid][category])
            del bad['computations'][cid]
            with self.assertRaises(ValueError):reconcile(self.records,bad)

    def test_wrong_category_and_result_rejected(self):
        for change in ('category','result'):
            bad=deepcopy(self.provenance);cid=self.records[0]['arms']['low']['reference']['candidates'][0]['computation_id']
            if change=='category':bad['computations'][cid]['oracle']=True
            else:bad['results'][cid]['value']['gap']=.9
            with self.assertRaises(ValueError):reconcile(self.records,bad)

    def test_L_shared_cached_results_are_traced(self):
        row=self.records[0]['arms']['low'];grid=next(c for c in row['reference']['candidates'] if c['source']=='grid')
        fine=next(c for c in row['reference']['candidates'] if c['source']=='fine_grid' and (c['u'],c['beta'])==(grid['u'],grid['beta']))
        self.assertEqual(grid['computation_id'],fine['computation_id'])
        self.assertNotEqual(grid['request_id'],fine['request_id'])
        self.assertTrue(self.provenance['resolutions'][fine['request_id']]['cached'])
        reconcile(self.records,self.provenance)

    def test_N_valid_full_synthetic_fixture(self):
        self.assertEqual(evidence.verify_reference(self.records[0]['arms']['low'],0,0),(.2,False))
        reconcile(self.records,self.provenance)
        self.assertGreater(storage.verify_ledger(self.root,'resources-final.json')['counters']['actual_evaluations'],8385)

    def test_M_real_interruption_retains_charged_partial_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);budget=Budget(root/'evaluations.jsonl',arm=1,cpu_clock=lambda:0.)
            obj=base.WorkflowTests().scalar_objective(budget);search=ReferenceSearch(obj,0,0)
            from struct2circuit.calibration.reference import ReferenceResult
            result=ReferenceResult('B00')
            def solver(fun):fun([.2,.3]);fun([.4,.5])
            with self.assertRaises(ArmExhausted):search._attempt(result,BY_NAME['B00'],'A','lbfgsb',solver,250,start=[.2,.3])
            record=result.record();self.assertEqual(len(record['candidates']),1)
            self.assertEqual(record['attempts'][0]['interrupted'],'ArmExhausted')
            self.assertEqual(record['attempts'][0]['objective_calls'],2)
            self.assertEqual(record['attempts'][0]['actual_evaluations'],1)
            self.assertFalse(evidence.verify_attempts(json.loads(json.dumps(record)),obj.identity))
            storage.checkpoint(root,budget,'final');budget.close()
            self.assertEqual(storage.verify_ledger(root,'resources-final.json')['counters']['actual_evaluations'],1)

    def test_full_call_cap_keeps_reference_flag_unresolved(self):
        from struct2circuit.calibration.reference import ReferenceResult
        with tempfile.TemporaryDirectory() as d:
            budget=Budget(Path(d)/'evaluations.jsonl',cpu_clock=lambda:0.)
            obj=base.WorkflowTests().scalar_objective(budget);search=ReferenceSearch(obj,0,0)
            result=ReferenceResult('B00',faces_complete=True)
            def runner(fun):
                for _ in range(251):fun([.2,.3])
            search._attempt(result,BY_NAME['B00'],'A','lbfgsb',runner,250,start=[.2,.3])
            self.assertEqual(result.attempts[0]['objective_calls'],250)
            self.assertEqual(len(result.attempts[0]['observations']),250)
            self.assertEqual(result.attempts[0]['interrupted'],'AttemptLimit')
            # Agreement with a synthetic second-route scalar cannot erase an interruption.
            result.candidates.append(replace(result.candidates[0],route='B'))
            search._pool(result,BY_NAME['B00'],())
            self.assertEqual(result.r,0.);self.assertFalse(result.complete)
            self.assertEqual(result.unresolved_reason,'mandatory_attempt_interrupted')
            budget.close()

    def test_backend_failure_remains_charged(self):
        from struct2circuit.alignment_experiment.reference import Objective
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);budget=Budget(root/'evaluations.jsonl',cpu_clock=lambda:0.)
            p=toy();sim=SimpleNamespace(problem=p,basis=p.feasible_basis(),mixer_hamiltonian=np.eye(20),
                initial_state=np.full(20,1/np.sqrt(20)),normalized_costs=np.linspace(0.,1.,20))
            sim.state_dimensionless=lambda *_:(_ for _ in ()).throw(ArithmeticError('synthetic backend failure'))
            obj=Objective(sim,budget,'0:0:low')
            with self.assertRaises(ArithmeticError):obj.evaluate(.2,.3)
            self.assertEqual(budget.counts['actual_evaluations'],1);budget.close()
            p=load_provenance(root);self.assertEqual(next(iter(p['results'].values()))['status'],'failed')

    def test_CPU_cutoff_after_computation_preserves_result_and_charge(self):
        from struct2circuit.alignment_experiment.reference import Objective
        from struct2circuit.calibration.reference import ReferenceResult
        from struct2circuit.calibration.budget import BudgetExhausted
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);clock=[0.];budget=Budget(root/'evaluations.jsonl',cpu_clock=lambda:clock[0])
            p=toy();state=np.full(20,1/np.sqrt(20),dtype=complex)
            def backend(*_):clock[0]=3300.;return state.copy()
            sim=SimpleNamespace(problem=p,basis=p.feasible_basis(),mixer_hamiltonian=np.eye(20),initial_state=state,
                normalized_costs=np.linspace(0.,1.,20),feasible_span=2.,_mixer_eigenvalues=np.arange(20),
                state_dimensionless=backend,_evaluate_state=lambda _:SimpleNamespace(normalized_gap=.2,expectation=.4,cost_status='nonconstant'))
            obj=Objective(sim,budget,'0:0:low');search=ReferenceSearch(obj,0,0);result=ReferenceResult('B00')
            with self.assertRaises(BudgetExhausted):search._attempt(result,BY_NAME['B00'],'A','lbfgsb',lambda fun:fun([.2,.3]),250,start=[.2,.3])
            record=json.loads(json.dumps(result.record()))
            self.assertFalse(evidence.verify_attempts(record,obj.identity))
            self.assertIsNotNone(record['attempts'][0]['unreturned_request_id'])
            budget.close();prov=load_provenance(root)
            self.assertEqual(next(iter(prov['results'].values()))['status'],'success')
            reconcile([dict(family=0,replicate=0,arms=dict(low=dict(bindings=dict(objective_identity=obj.identity),reference=record,controls=[])))],prov)

    def test_full_verifier_accepts_bound_synthetic_evidence_and_rejects_empty_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);impl=config.identity();records=unknown_rows();p=toy();selected=selection.select(p)
            row=records[0];row.update(Q=p.Q.tolist(),c=p.c.tolist(),selection=selected)
            budget=Budget(root/'evaluations.jsonl',cpu_clock=lambda:0.)
            for arm,g in selected['selected'].items():
                a=scalar_arm();identity=evidence.bindings(p,g['edges'],f'0:0:{arm}',impl);a['bindings']=identity
                h=xy_mixer_hamiltonian(p.feasible_basis(),MixerSpec('c6',6,g['edges'],'synthetic'))
                width=float(np.ptp(np.linalg.eigh(h)[0]));pts=interior(0,0,width).tolist()
                mapping=dict(zip(map(tuple,a['interior_points']),pts));a.update(interior_points=pts,mixer_width=width)
                perm=(np.arange(6)+1)%6;inv=np.argsort(perm)
                transport=CardinalityQUBO(p.Q[np.ix_(inv,inv)],p.c[inv],3)
                edges=tuple(tuple(sorted((int(perm[i]),int(perm[j])))) for i,j in g['edges'])
                tid=evidence.bindings(transport,edges,f'0:0:{arm}',impl)['objective_identity']
                for check in a['controls']:
                    check['bindings']=identity
                    if tuple(check['point']) in mapping:check['point']=mapping[tuple(check['point'])]
                    if check['kind']=='gradient':check['steps']=[1e-5,1e-5/max(1.,width)]
                    if check['kind']=='cyclic_transport':check['transported_identity']=tid
                row['arms'][arm]=materialize(a,budget,identity['objective_identity'],f'0:0:{arm}')
            auth=dict(execution_authorized=True,scope=config.SCOPE,design_sha256=config.DESIGN_HASH,
                implementation=impl,manifest_sha256=canonical_hash(config.manifest()),output=str(root),
                approval_identity='SYNTHETIC-TEST-ONLY',historical_archive='/workspace/calibration-archives/stage7b-development-e8dea4d-v1.tar.gz',
                historical_archive_sha256=config.HISTORY_HASH,
                development_records={str(Path('/workspace/calibration-runs')/n/'instances.jsonl'):h for n,h in config.DEVELOPMENT.items()})
            atomic_json(root/'authorization.json',auth)
            storage.finish(root,budget,records,impl,storage.file_hash(root/'authorization.json'),previous=None);budget.close()
            atomic_json(root/'worker-ended.json',dict(cpu_seconds=.1,process_ended=True));storage.settle(root,.2)
            self.assertEqual(storage.verify(root)['analysis']['counts']['neutral'],1)
            # Internally consistent zero receipts still cannot support the same references.
            (root/'evaluations.jsonl').unlink();(root/'evaluations.results.jsonl').unlink()
            (root/'resources-final.json').unlink()
            zero=Budget(root/'evaluations.jsonl',cpu_clock=lambda:0.)
            storage.checkpoint(root,zero,'final');zero.close()
            inventory=json.loads((root/'inventory.json').read_text())
            for name in ('evaluations.jsonl','evaluations.results.jsonl','resources-final.json'):
                inventory['files'][name]=storage.file_hash(root/name)
            (root/'inventory.json').write_text(json.dumps(inventory))
            settled=json.loads(Path(str(root)+'.settled.json').read_text())
            settled['inventory_sha256']=storage.file_hash(root/'inventory.json')
            Path(str(root)+'.settled.json').write_text(json.dumps(settled))
            legacy={}
            for name in ('evidence','storage'):
                source=subprocess.check_output(['git','show',f'8db4d2d30ad56df3249839455cbd57f5eeca02db:src/struct2circuit/alignment_experiment/{name}.py'],cwd=config.ROOT,text=True)
                namespace={'__package__':'struct2circuit.alignment_experiment'}
                exec(compile(source,f'reviewed-{name}.py','exec'),namespace);legacy[name]=namespace
            legacy['storage']['analyze']=legacy['evidence']['analyze']
            self.assertEqual(legacy['storage']['verify'](root)['analysis']['counts']['neutral'],1)
            with self.assertRaisesRegex(ValueError,'request'):storage.verify(root)


if __name__=='__main__':unittest.main()
