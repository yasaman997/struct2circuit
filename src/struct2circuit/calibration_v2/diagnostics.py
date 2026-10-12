"""Optional wider searches on persisted models; immutable contradiction addenda."""
from dataclasses import asdict, replace
from pathlib import Path
import json

import numpy as np

from ..calibration import config as v1
from ..calibration.budget import BudgetExhausted
from ..calibration.decisions import (GAIN, agreement, band, challenge_required, deficit, ordering,
                                    reference_status, reversal)
from ..calibration.diagnostics import direct_oracle
from ..calibration.driver import Case, CaseResult
from ..calibration.objective import Objective, array_hash
from ..calibration.reference import Candidate, ReferenceResult, ReferenceSearch
from ..calibration.storage import OutputStore
from ..mixers import MixerSpec
from ..problems import CardinalityQUBO
from ..simulator import FeasibleSubspaceQAOA
from .budget import Budget
from .config import PROTOCOL_HASH, identity, manifest
from .driver import PhaseDriver
from .evidence import assess
from .storage import (atomic_json, authorize, current_snapshot, file_hash, finish_resources,
                      runtime_identity, verify_evidence)

DOMAINS=('B10','B01','B11','B22','Bsign')


def contradiction_status(original,*,reliable=False,inside=False,recalculated=None,shared_failure=False,unchecked=False):
    if original=='FAILED':return 'FAILED'
    if shared_failure:return 'FAILED'
    if unchecked:return 'INCONCLUSIVE'
    if reliable and inside and recalculated in ('FAILED','INCONCLUSIVE'):return recalculated
    return original


def addendum(path,*,cores,lock_sha256,evidence,effective_status,budget):
    from ..calibration.storage import canonical_hash
    payload=dict(schema_version=2,protocol_sha256=PROTOCOL_HASH,core_artifacts=cores,
        lock_sha256=lock_sha256,evidence=evidence,effective_status=effective_status,
        resources=budget.snapshot(),scope='P33/B00; later diagnostics are exploratory',
        policy_substitution_authorized=False,validation_retry_authorized=False)
    return atomic_json(path,dict(payload=payload,sha256=canonical_hash(payload)))


def restore_reference(record):
    record=record.copy();record['candidates']=[Candidate(**c) for c in record['candidates']]
    return ReferenceResult(**record)


def restore_models(directory,budget):
    """Load exact retained arrays, never call a QUBO generator or choose new states."""
    p=Path(directory)
    instances={r['instance_id']:r for r in map(json.loads,(p/'instances.jsonl').read_text().splitlines())}
    rows={tuple(r['landscape']):r for r in map(json.loads,(p/'landscapes.jsonl').read_text().splitlines())}
    cases=[];results={}
    for graph in map(json.loads,(p/'graphs.jsonl').read_text().splitlines()):
        key=tuple(graph['landscape']);i,g,init=key;row=rows[key];instance=instances[i]
        canonical=tuple(row['canonical'])
        if canonical in results:
            sim=results[canonical].case.simulator
        else:
            budget.check_time()
            problem=CardinalityQUBO(np.asarray(instance['Q']),np.asarray(instance['c']),instance['k'])
            mixer=MixerSpec('persisted',problem.n,tuple(map(tuple,graph['edges'])),'persisted v2')
            sim=FeasibleSubspaceQAOA(problem,mixer,initialization=v1.INITIALIZATIONS[init])
            budget.counts['eigensolver_calls']+=1
            sim.initial_state=np.asarray(graph['initial_state_real'])+1j*np.asarray(graph['initial_state_imag'])
            if array_hash(sim.initial_state)!=graph['initial_state_sha256']:
                raise ValueError('Persisted state identity changed')
        case=Case(i,instance['family'],g,init,canonical[1],sim,row['n_beta'],row['audit'],
                  (g,init)==v1.SENTINELS[instance['family']],instance['split'],tuple(graph['shuffle_permutation']))
        cases.append(case)
        if canonical in results:
            results[canonical].case=replace(results[canonical].case,
                audit=results[canonical].case.audit or case.audit,sentinel=results[canonical].case.sentinel or case.sentinel)
        if canonical not in results:
            obj=Objective(sim,budget,':'.join(map(str,canonical)),optional=True)
            if obj.identity!=row['objective_identity']:raise ValueError('Persisted objective identity changed')
            search=ReferenceSearch(obj,case.n_beta,canonical)
            search.trace_cache={tuple(t['key']):([Candidate(**c) for c in t['candidates']],t['done'],t['flat']) for t in row['trace_cache']}
            results[canonical]=CaseResult(case,obj,search,
                references={n:restore_reference(r) for n,r in row['references'].items()},
                production={int(n):Candidate(**c) for n,c in row['production'].items()},
                controls=row['controls'],initial_metrics=row['initial_metrics'])
    return cases,results


class DiagnosticDriver:
    def __init__(self,cases,results,budget,store):
        self.cases,self.results,self.budget,self.store=cases,results,budget,store
        self.not_run=[];self.contradictions=[]
        self.required_challenges={r.case.canonical_key for r in results.values() if r.case.sentinel}
        self.ordered=sorted(results.values(),key=lambda r:(r.case.initialization_id,r.case.graph_id,r.case.instance_id))

    def admit(self,result,domain,level):
        # An inventory of past keys is not a cache of state/objective values.
        axes=v1.grid_axes(domain,result.case.n_beta,level)
        fresh=sum(result.objective.key(u,b) not in result.objective.cache for u in axes[0] for b in axes[1])
        remaining=min(self.budget.limits.total-self.budget.counts['actual_evaluations'],
                      self.budget.limits.landscape-self.budget.landscapes[result.objective.landscape])
        if fresh>remaining:
            self.not_run.append(dict(landscape=result.case.key,domain=domain.name,level=level,
                reason='minimum_fresh_grid_exceeds_remaining_allowance',status='NOT_RUN',adequacy='INCONCLUSIVE'))
            return False
        self.budget.check_time()
        if self.budget.resource_guard:self.budget.resource_guard.check(self.budget,force=True)
        return True

    def search(self,result,name,action='search',audit_only=False):
        domain=v1.BY_NAME[name]
        if not self.admit(result,domain,0 if action=='search' else 1):return False
        ref=result.references.get(name)
        contained=tuple(c for n,r in result.references.items() if n!=name and domain.includes(v1.BY_NAME[n]) for c in r.candidates)
        try:
            if action=='search':result.references[name]=result.search.search(domain,contained=contained)
            elif action=='refine':result.search.refine(ref,contained=contained,audit_only=audit_only)
            else:result.search.complete_bundle(ref,domain)
            self.inspect_contradiction(result)
            return True
        except BudgetExhausted as error:
            self.not_run.append(dict(landscape=result.case.key,domain=name,status='INCONCLUSIVE',reason=str(error)))
            return False
        except (ArithmeticError,ValueError) as error:
            self.contradictions.append(dict(landscape=result.case.key,shared_failure=True,
                source_domain=name,error=str(error)))
            return False

    def inspect_contradiction(self,result):
        core=result.references['B00']
        better=[c for n,r in result.references.items() if n!='B00' for c in r.candidates
                if v1.BY_NAME['B00'].contains(c.point) and c.value<core.pooled]
        if not better:return
        candidate=min(better,key=lambda c:c.value)
        event=dict(landscape=result.case.key,candidate=asdict(candidate),inside_B00=True,reliable=False)
        # Append before checks: exhaustion must not conceal the observation.
        self.contradictions.append(event)
        try:
            native=result.objective.evaluate(*candidate.point,'B00',fresh=True)
            def record(c):
                event.update(oracle=c)
                if c['status']=='fail':event['shared_failure']=True
            check=direct_oracle(result.objective,candidate.point,record_callback=record)
            event.update(oracle=check,reliable=check['status']=='pass' and abs(native.gap-candidate.value)<=1e-10)
            result.controls.append(check)
            result.controls.append(dict(kind='candidate_reevaluation',point=candidate.point,domain='B00',
                error=abs(native.gap-candidate.value),status='pass' if abs(native.gap-candidate.value)<=1e-10 else 'fail'))
            if not event['reliable']:
                event['shared_failure']=True;return
            result.search._pool(core,v1.BY_NAME['B00'],(candidate,))
            if not core.refined and self.admit(result,v1.BY_NAME['B00'],1):
                result.search.refine(core,production=result.production[33])
            elif core.refined and not core.full_bundle:
                result.search.complete_bundle(core,v1.BY_NAME['B00'],production=result.production[33])
        except BudgetExhausted:
            event['unchecked']=True
        except (ArithmeticError,ValueError) as error:
            event.update(shared_failure=True,error=str(error))

    def run(self):
        # All routine coverage precedes optional refinements; no production rerun.
        for result in self.ordered:
            for name in DOMAINS[:3]:
                if not self.budget.stop_reason:self.search(result,name)
        for result in self.ordered:
            if result.case.sentinel:
                for name in DOMAINS[3:]:
                    if not self.budget.stop_reason:self.search(result,name)
        for result in self.ordered:
            if result.case.audit:
                for name in DOMAINS[:3]:
                    if name in result.references and not self.budget.stop_reason:self.search(result,name,'refine',True)
        for _ in range(1+14*len(self.ordered)):
            challenge=set()
            for result in self.ordered:
                r11=result.references.get('B11');r00=result.references['B00']
                near=bool(r11 and r11.selected and any(x<=.01 for x in v1.BY_NAME['B11'].face_distances(r11.selected.point)[1]))
                if challenge_required(sentinel=result.case.sentinel,near_upper=near,
                    gain=deficit(r00.pooled,r11.pooled) if r11 and r11.complete else None):challenge.add(result.case.canonical_key)
            from itertools import combinations
            for i,init in sorted({(c.instance_id,c.initialization_id) for c in self.cases}):
                group=[r for r in self.ordered if (r.case.instance_id,r.case.initialization_id)==(i,init)]
                for a,b in combinations(group,2):
                    for first,second in combinations(('B00','B10','B01','B11'),2):
                        refs=[r.references.get(n) for r in (a,b) for n in (first,second)]
                        if all(r and r.complete for r in refs):
                            change=reversal(refs[2].pooled-refs[0].pooled,refs[3].pooled-refs[1].pooled)
                            if change in ('reversal','ambiguous'):
                                challenge.update((a.case.canonical_key,b.case.canonical_key))
                            if change=='ambiguous':
                                a.ranking_boxes.update((first,second));b.ranking_boxes.update((first,second))
            self.required_challenges.update(challenge)
            changed=False
            for result in self.ordered:
                if self.budget.stop_reason:break
                for name in DOMAINS:
                    ref=result.references.get(name)
                    if ref is None:
                        if name in DOMAINS[3:] and result.case.canonical_key in challenge:
                            changed |= self.search(result,name)
                            break
                        continue
                    baseline=result.references['B00' if name!='Bsign' else 'B11']
                    reason=not ref.complete or band(deficit(baseline.pooled,ref.pooled),GAIN)!='pass' or name in result.ranking_boxes
                    if ref.selected:
                        reason |= any(d<=.01 and f'{axis}:{fixed}' not in ref.flat_faces
                            for axis,distances in enumerate(zip(*v1.BY_NAME[name].face_distances(ref.selected.point)))
                            for fixed,d in zip(v1.BY_NAME[name].bounds[axis],distances))
                    if reason and not ref.refined:
                        changed |= self.search(result,name,'refine');break
                    if reason and not ref.full_bundle:
                        changed |= self.search(result,name,'bundle');break
            if not changed or self.budget.stop_reason:break
        return self

    def status(self):
        gains=[];unresolved=bool(self.not_run)
        for result in self.ordered:
            for name in DOMAINS:
                if name in DOMAINS[3:] and result.case.canonical_key not in self.required_challenges:continue
                ref=result.references.get(name);base=result.references.get('B11' if name=='Bsign' else 'B00')
                if not ref or not ref.complete or not base or not base.complete:unresolved=True;continue
                gains.append(band(deficit(base.pooled,ref.pooled),GAIN))
        from itertools import combinations
        for i,init in sorted({(c.instance_id,c.initialization_id) for c in self.cases}):
            group=[r for r in self.ordered if (r.case.instance_id,r.case.initialization_id)==(i,init)]
            for a,b in combinations(group,2):
                for first,second in combinations(('B00','B10','B01','B11'),2):
                    refs=[r.references.get(n) for r in (a,b) for n in (first,second)]
                    if all(r and r.complete for r in refs):
                        change=reversal(refs[2].pooled-refs[0].pooled,refs[3].pooled-refs[1].pooled)
                        if change=='reversal':gains.append('fail')
                        if change=='ambiguous':unresolved=True
        return 'DEFEATED' if 'fail' in gains else 'INCONCLUSIVE' if unresolved or 'ambiguous' in gains else 'ADEQUATE'


def run_diagnostics(output,*,development,validation,resource,lock,authorization,protocol,cpu_origin=None):
    from .config import verify_protocol
    from .storage import ValidationGate
    verify_protocol(protocol);auth=authorize(authorization,'diagnostics',output)
    runtime_identity();gate=ValidationGate(lock)
    if gate.payload.get('diagnostics_authorized_before_validation') is not True:
        raise PermissionError('The optional diagnostic study was not authorized before validation')
    hashes={str(Path(p).absolute()):verify_evidence(p,phase=phase) for p,phase in
            ((development,'development'),(validation,'validation'))}
    if hashes!=auth['core_artifact_hashes']:raise PermissionError('Diagnostic core identities do not match authorization')
    m=json.loads((Path(validation)/'calibration_manifest.json').read_text())
    if m.get('validation_lock_sha256')!=gate.bytes_hash:raise ValueError('Validation used a different lock')
    b=Budget.from_snapshot(current_snapshot(validation,resource),phase='diagnostics',cpu_origin=cpu_origin)
    from ..calibration.resources import ResourceGuard
    b.resource_guard=ResourceGuard(output);b.resource_guard.check(b,force=True)
    from ..calibration.storage import preflight_validation_lock
    for p in (str(output)+'.resources.json',str(output)+'.resources.json.process-ended.json'):
        preflight_validation_lock(p)
    with OutputStore(output) as store:
        store.json('calibration_manifest.json',manifest()|dict(phase='diagnostics',core_artifacts=hashes,authorization=auth,status='RUNNING'))
        cases=[];results={}
        for path in (development,validation):
            c,r=restore_models(path,b);cases+=c;results.update(r)
        diagnostic=DiagnosticDriver(cases,results,b,store).run()
        effective='FAILED' if any(e.get('shared_failure') for e in diagnostic.contradictions) else 'QUALIFIED'
        if diagnostic.contradictions:
            # Recalculate affected original panels using pooled B00 evidence and
            # preserve original controls. Newly selected points need their checks.
            rows=[]
            for case in cases:
                r=results[case.canonical_key]
                rows.append(dict(landscape=case.key,canonical=case.canonical_key,status='observed',audit=case.audit,
                    width=float(np.ptp(case.simulator._mixer_eigenvalues)),production={str(n):asdict(c) for n,c in r.production.items()},
                    references={'B00':r.references['B00'].record()},controls=r.controls,synthetic=False,error=r.error))
            recalculated=assess(rows,[c.key for c in cases])[0]
            for e in diagnostic.contradictions:
                effective=contradiction_status(effective,reliable=e.get('reliable'),inside=e.get('inside_B00'),
                    recalculated=recalculated,shared_failure=e.get('shared_failure'),unchecked=e.get('unchecked'))
            addendum(Path(output)/'contradiction-addendum.json',cores=hashes,lock_sha256=gate.bytes_hash,
                evidence=diagnostic.contradictions,effective_status=effective,budget=b)
        for case in cases:
            r=results[case.canonical_key]
            store.row('landscapes.jsonl',dict(landscape=case.key,references={n:ref.record() for n,ref in r.references.items()},
                                            optional=True))
        store.json('budget.json',b.snapshot())
        store.decision(f'Diagnostic adequacy: {diagnostic.status()}\nEffective core status: {effective}\nExploratory; no expanded production qualification.')
        store.json('calibration_manifest.json',manifest()|dict(phase='diagnostics',status=diagnostic.status(),effective_core_status=effective,
            not_run=diagnostic.not_run,core_artifacts=hashes,authorization=auth),update_owned=True);store.sync()
    finish_resources(output,b,'diagnostics',previous=resource)
    return effective
