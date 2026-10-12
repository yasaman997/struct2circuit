"""B00-first scheduling, sharing the unchanged v1 numerical algorithms."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import os
from time import process_time

import numpy as np

from .. import problems
from ..calibration import config as v1
from ..calibration.budget import BudgetExhausted
from ..calibration.driver import Case, CaseResult, PhaseDriver as V1Driver, _points
from ..calibration.diagnostics import (direct_oracle, gradient_check, invariance_checks,
    reuse_initialization, spectral_coordinates, spectral_diagnostics)
from ..calibration.objective import Objective, array_hash, production_search
from ..calibration.reference import Candidate
from .reference import ReferenceSearch
from ..calibration.resources import ResourceGuard, ResourceExhausted
from ..calibration.storage import OutputStore, ARTIFACTS, PAIR_FIELDS, provenance, preflight_validation_lock
from ..simulator import FeasibleSubspaceQAOA
from .budget import Budget
from .config import PROTOCOL_HASH, SCOPE, expected_keys, identity, manifest, specs, verify_protocol
from .evidence import assess, required_work


class PhaseDriver(V1Driver):
    def __init__(self, cases, budget, store, *, expected_keys, synthetic=False, validation_gate=None):
        super().__init__(cases,budget,store,expected_keys=expected_keys,policies=(33,),
                         synthetic=synthetic,validation_gate=validation_gate)
        self.counterexamples={}

    def add_policy(self, policy):
        raise ValueError('P33 is fixed prospectively; no fallback or escalation policy')

    def _routine(self, case):
        obj=Objective(case.simulator,self.budget,':'.join(map(str,case.canonical_key)))
        search=ReferenceSearch(obj,case.n_beta,case.canonical_key,
            candidate_callback=lambda b,c,e:self._emit_candidate(case.key,b,c,e))
        result=CaseResult(case,obj,search);self.results[case.canonical_key]=result
        if case.simulator.cost_status != 'nonconstant':
            result.error='constant_or_unresolved';return
        try:
            initial=obj.evaluate(0.,0.)
            result.initial_metrics=initial.record() | dict(initial_optimum_probability=float(
                np.sum(abs(initial.state[case.simulator.optimal_mask])**2)))
            interior,corners,control=_points(case)
            for point in interior:
                self._run_control(result,direct_oracle,obj,point,'B11')
                self._record_control(result,gradient_check(obj,point,'B11'))
            if result.arithmetic_failure:
                raise ArithmeticError('Pre-search numerical control failed')
            optimized,value=production_search(obj,33)
            result.production[33]=Candidate(value.u,value.beta,value.gap,'production','P33',objective_identity=obj.identity)
            self.store.row('candidates.jsonl',dict(landscape=case.key,domain='B00',policy=33,
                evaluation=value.record(),optimizer=asdict(optimized),accounting=dict(self.budget.counts)))
            result.references['B00']=search.search(v1.BY_NAME['B00'],production=result.production[33])
            if self.budget.stop_reason:raise BudgetExhausted(self.budget.stop_reason)
            for point in corners:self._run_control(result,direct_oracle,obj,point,'B11')
            for point in control:self._run_control(result,spectral_coordinates,obj,point)
            if case.instance_id in (0,2,4,6) or self.synthetic:
                invariance_checks(obj,control,record_callback=lambda c:self._record_control(result,c))
            self._selected_oracles(result)
            self._capture_failures()
        except BudgetExhausted as error:result.error=str(error)
        except (ArithmeticError,ValueError) as error:
            result.error=str(error);result.arithmetic_failure=True

    def row(self, case):
        result=self.results.get(case.canonical_key)
        row=dict(landscape=case.key,canonical=case.canonical_key,status='observed' if result else 'missing',
            audit=case.audit,n_beta=case.n_beta,width=float(np.ptp(case.simulator._mixer_eigenvalues)),
            synthetic=self.synthetic,production={},references={},controls=[],error=None)
        if result:
            row.update(production={str(n):asdict(c) for n,c in result.production.items()},
                references={n:r.record() for n,r in result.references.items()},controls=result.controls,
                error=result.error,arithmetic_failure=result.arithmetic_failure,
                objective_identity=result.objective.identity,initial_metrics=result.initial_metrics,
                cost_status=case.simulator.cost_status,feasible_span=case.simulator.feasible_span,
                family=case.family,split=case.split,n=case.simulator.problem.n,k=case.simulator.problem.k)
            row['negative_evidence']=[v for v in self.counterexamples.values() if
                tuple(v['canonical'])==case.canonical_key]
        return row

    def _capture_failures(self):
        from ..calibration.decisions import agreement,band,DEFICIT,DISTORTION,distortion
        from .evidence import reference_values
        values={}
        for key,result in self.results.items():
            ref=result.references.get('B00');p=result.production.get(33)
            if ref is None or p is None:continue
            r,complete=reference_values(ref.record())
            if not complete:continue
            proof=dict(a=ref.route_a,b=ref.route_b,r=r,faces_complete=ref.faces_complete,
                       refined=ref.refined,coarse=ref.coarse,production=p.value)
            values[key]=proof
            if band(p.value-r,DEFICIT)=='fail':
                self.counterexamples.setdefault(('E',key),dict(kind='E',canonical=key,values=[proof]))
        from itertools import combinations
        for left,right in combinations(sorted(values),2):
            if (left[0],left[2])!=(right[0],right[2]):continue
            a,b=values[left],values[right]
            if band(distortion(a['production'],b['production'],a['r'],b['r']),DISTORTION)=='fail':
                self.counterexamples.setdefault(('K',left,right),dict(kind='K',canonical=left,
                    other=right,values=[a,b]))

    def assessment(self):
        rows=[self.row(c) for c in self.cases]
        return assess(rows,self.expected_keys,stopped=bool(self.budget.stop_reason),phase_failures=self.phase_failures)

    def _paired_rows(self):
        self._capture_failures()
        _,_,self.pairs=self.assessment()

    def _scheduled_work(self):
        for result in self.results.values():
            if result.error or not result.case.audit:continue
            try:
                self._capture_failures()
                result.search.refine(result.references['B00'],production=result.production[33],audit_only=True)
                self._selected_oracles(result)
            except BudgetExhausted as error:result.error=str(error)
            except (ArithmeticError,ValueError) as error:
                result.error=str(error);result.arithmetic_failure=True

    def _settle(self):
        ordered=sorted(self.results.values(),key=lambda r:(r.case.initialization_id,r.case.graph_id,r.case.instance_id))
        for _ in range(1+2*len(ordered)):
            self._paired_rows();changed=False
            triggers={}
            for p in self.pairs:
                if p['status'] != 'pass':
                    for g in (p['i'],p['j']):
                        case=next(c for c in self.cases if c.key==(p['instance_id'],g,p['initialization_id']))
                        triggers[case.canonical_key]=True
            for result in ordered:
                if result.error or not required_work(self.row(result.case),triggers.get(result.case.canonical_key,False)):continue
                ref=result.references.get('B00')
                if ref is None:continue
                try:
                    if not ref.refined:
                        result.search.refine(ref,production=result.production[33])
                    elif not ref.full_bundle:
                        result.search.complete_bundle(ref,v1.BY_NAME['B00'],production=result.production[33])
                    else:continue
                    changed=True;self._selected_oracles(result)
                except BudgetExhausted as error:result.error=str(error)
                except (ArithmeticError,ValueError) as error:
                    result.error=str(error);result.arithmetic_failure=True
            if self.budget.stop_reason or not changed:break
        self._paired_rows()

    def status(self,policy=33):
        if policy != 33:raise ValueError('Only P33 may qualify')
        return self.assessment()[0]

    def checks(self,policy=33,*,observe=None):
        if policy != 33:raise ValueError('Only P33 may qualify')
        _,checks,_=self.assessment()
        if observe:
            for c in checks:observe(c)
        return checks,[c.name for c in checks]

    def write_results(self,policy=33,*,final_status=None):
        status,checks,self.pairs=self.assessment()
        if final_status and status != 'FAILED':status=final_status
        # Release state caches before large terminal JSON snapshots. Diagnostics
        # restore models, never treat a unique-key inventory as a value cache.
        for r in self.results.values():r.objective.release_cache()
        if self.budget.resource_guard:self.budget.resource_guard.check(self.budget,force=True)
        for case in self.cases:
            sim=case.simulator;result=self.results.get(case.canonical_key)
            self.store.row('graphs.jsonl',dict(landscape=case.key,canonical=case.canonical_key,
                graph_label=v1.GRAPH_LABELS[case.graph_id],edges=sim.mixer.edges,
                spectrum=spectral_diagnostics(sim),n_beta=case.n_beta,
                initial_state_real=sim.initial_state.real,initial_state_imag=sim.initial_state.imag,
                initial_state_sha256=array_hash(sim.initial_state),edge_hash=array_hash(np.asarray(sim.mixer.edges)),
                shuffle_permutation=case.shuffle_permutation,initial_metrics=result.initial_metrics if result else None))
            row=self.row(case)
            if result:
                row['trace_cache']=[dict(key=k,candidates=[asdict(c) for c in v[0]],done=v[1],flat=v[2])
                                    for k,v in result.search.trace_cache.items()]
            self.store.row('landscapes.jsonl',row)
        for key in self.expected_keys:
            if key not in {c.key for c in self.cases}:
                self.store.row('landscapes.jsonl',dict(landscape=key,status='missing'))
        for pair in self.pairs:self.store.pair({n:pair.get(n) for n in PAIR_FIELDS})
        self.budget.mandatory_complete=status=='QUALIFIED'
        self.store.json('budget.json',self.budget.snapshot() | dict(incomplete_required_work=[c.name for c in checks if c.status!='pass']))
        statistics,_=self.descriptive_records(33)
        self.store.decision(f'{status}\nPhase numerical status: {status}.\nPolicy: P33 on B00 only.\nScope: {SCOPE}.\n'
            f'Checks: {json.dumps([asdict(c) for c in checks])}\nDescriptive E/K: {json.dumps(statistics)}\n'
            'Development qualification permits lock eligibility only. Whole-study qualification requires validation.\n'
            'Historical P9/P17 remain FAILED/ineligible. Expanded-domain adequacy: NOT TESTED.\n'
            'No RMS superiority or global optimality. Benchmark: DRAFT_UNFROZEN.')
        return status


def generate_cases(phase,budget,gate=None,*,authorization,output):
    from .storage import authorize
    authorize(authorization,phase,output,lock=gate.path if gate else None)
    if phase=='validation':
        if gate is None:raise PermissionError('A consumed v2 gate is required')
        gate.begin_generation(33)
    cases=[]
    for spec in specs(phase):
        budget.check_time()
        generator,parameters=v1.FAMILIES[spec.family]
        problem=getattr(problems,generator)(n=spec.n,k=spec.k,seed=spec.seed,**parameters)
        graphs,canonical,permutation=v1.construct_graphs(problem,spec.instance_id)
        bases={}
        for g in sorted(set(canonical)):
            bases[g]=FeasibleSubspaceQAOA(problem,graphs[g]);budget.counts['eigensolver_calls']+=1;budget.check_time()
        n_beta=v1.beta_intervals(max(np.ptp(s._mixer_eigenvalues) for s in bases.values()))
        for g in range(8):
            for init,name in enumerate(v1.INITIALIZATIONS):
                cases.append(Case(spec.instance_id,spec.family,g,init,canonical[g],
                    reuse_initialization(bases[canonical[g]],name,budget),n_beta,
                    g==v1.AUDIT_GRAPHS[spec.family],(g,init)==v1.SENTINELS[spec.family],spec.split,tuple(permutation)))
    return tuple(cases)


def run_phase(output,*,phase,authorization,protocol,lock=None,lock_output=None,cpu_origin=None):
    from .storage import ValidationGate, authorize, finish_resources, require_official_runtime
    verify_protocol(protocol)
    auth=authorize(authorization,phase,output,lock=lock)
    runtime=require_official_runtime()
    guard=ResourceGuard(output,lock_output or lock)
    guard.check(force=True,extra_memory=8*Path(lock).stat().st_size if lock else 0)
    v1.verify_public_seed_metadata(Path(__file__).resolve().parents[3])
    gate=ValidationGate(lock) if phase=='validation' and lock else None
    if phase=='validation' and gate is None:raise PermissionError('Validation requires a v2 lock')
    if phase not in ('development','validation'):raise ValueError('Use the separate diagnostics entry path')
    budget=Budget.from_snapshot(gate.current_budget(),phase='validation',cpu_origin=cpu_origin) if gate else Budget(cpu_origin=cpu_origin)
    budget.resource_guard=guard;guard.check(budget,force=True)
    for path in (str(output)+'.resources.json',str(output)+'.resources.json.process-ended.json',
                 str(output)+'.resources.json.process-ended.json.settled.json'):
        preflight_validation_lock(path)
    if gate:preflight_validation_lock(str(gate.resource_path)+'.validation-consumed.json')
    if lock_output:
        if phase!='development':raise ValueError('Only development can publish a lock')
        preflight_validation_lock(lock_output)
        preflight_validation_lock(str(lock_output)+'.publication.json')
        preflight_validation_lock(str(lock_output)+'.publication-ended.json')
        if Path(lock_output).resolve().is_relative_to(Path(output).resolve()):
            raise FileExistsError('Lock must be outside immutable core output')
    with OutputStore(output) as store:
        m=manifest() | dict(phase=phase,status='RUNNING',expected_phase_keys=expected_keys(phase),
            provenance=provenance() | dict(driver=identity()),runtime=runtime,authorization=auth,
            validation_lock_sha256=gate.bytes_hash if gate else None,reporting_complete=False)
        store.json('calibration_manifest.json',m);driver=None
        try:
            cases=generate_cases(phase,budget,gate,authorization=authorization,output=output)
            for spec in specs(phase):
                sim=next(c.simulator for c in cases if c.instance_id==spec.instance_id)
                store.row('instances.jsonl',asdict(spec)|dict(status='generated',Q=sim.problem.Q,c=sim.problem.c,
                    coefficients_sha256=array_hash(sim.problem.Q,sim.problem.c),generator=v1.FAMILIES[spec.family][0],parameters=v1.FAMILIES[spec.family][1]))
            driver=PhaseDriver(cases,budget,store,expected_keys=expected_keys(phase),validation_gate=gate).run()
            status=driver.write_results()
            m.update(status=status,phase_status=status,reporting_complete=True,
                development_finalized=phase=='development' and status=='QUALIFIED',
                validation_completed=phase=='validation' and status=='QUALIFIED')
            store.json('calibration_manifest.json',m,update_owned=True);store.sync()
        except (BudgetExhausted,ArithmeticError,ValueError,MemoryError) as error:
            # Minimal recovery preserves failures already independently calculable.
            failed=bool(driver and (driver.counterexamples or driver.phase_failures or any(r.arithmetic_failure for r in driver.results.values())))
            if driver:
                try:failed |= driver.status()=='FAILED'
                except MemoryError:pass
            status='FAILED' if failed or isinstance(error,(ArithmeticError,ValueError)) else 'INCONCLUSIVE'
            m.update(status=status,error=str(error),reporting_complete=False)
            store.json('calibration_manifest.json',m,update_owned=True)
            store.decision(status+'\nReporting or required execution incomplete; no validation lock.\n'+str(error))
            store.sync()
    # Immutable resource sidecars carry finalization/publication overhead.
    record=finish_resources(output,budget,phase,previous=gate.grant_path if gate else None)
    # The supervisor may publish a lock only after it seals full worker CPU.
    return status
