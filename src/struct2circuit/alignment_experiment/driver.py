"""Single authorized mechanism run. No import-time generation or optimization."""
from dataclasses import fields
from pathlib import Path
import json
import os

import numpy as np

from .. import problems
from ..calibration.budget import BudgetExhausted
from ..calibration.objective import Evaluation, array_hash
from ..calibration.reference import Candidate, ReferenceResult
from ..calibration_v2.storage import atomic_json
from ..mixers import MixerSpec
from ..simulator import FeasibleSubspaceQAOA
from .budget import Budget, Guard
from .config import identity, panel, runtime, public_seed_check
from .evidence import bindings
from .reference import Objective, ReferenceSearch, pre_controls, transport_controls, final_controls, interior
from .selection import select
from .storage import authorization, claim, checkpoint, file_hash, finish


def _controls(row, records):
    for c in records:
        c['bindings'] = row['bindings']
        row['controls'].append(c)
        if c['status'] != 'pass':
            row['integrity_failure'] = True
            raise ArithmeticError('Essential numerical control failed')


def _cache_write(path, obj):
    entries = list(obj.cache.values())
    metadata = [dict(u=e.u, beta=e.beta, gamma=e.gamma, gap=e.gap, expectation=e.expectation,
                     span=e.span, cost_status=e.cost_status, mixer_width=e.mixer_width) for e in entries]
    with path.open('xb') as f:
        np.savez(f, metadata=json.dumps(metadata), identity=obj.identity,
                 states=np.array([e.state for e in entries]))
        f.flush(); os.fsync(f.fileno())
    obj.release_cache()


def _cache_read(path, digest, obj):
    if file_hash(path) != digest:
        raise ValueError('Coarse objective cache altered')
    with np.load(path, allow_pickle=False) as z:
        if str(z['identity']) != obj.identity:
            raise ValueError('Coarse cache belongs to another objective')
        metadata = json.loads(str(z['metadata']))
        states = z['states']
        for e, state in zip(metadata, states):
            state = state.copy(); state.setflags(write=False)
            obj.cache[obj.key(e['u'], e['beta'])] = Evaluation(**e, state=state)


def _result(record):
    value = dict(record)
    value['candidates'] = [Candidate(**c) for c in value['candidates']]
    return ReferenceResult(**value)


def _empty(spec):
    return dict(family=spec['family'], replicate=spec['replicate'], spec=spec, Q=None, c=None,
                selection=None, arms={a: dict(bindings=None, reference=None, controls=[], faces=[],
                unresolved_reason='not_executed', integrity_failure=False) for a in ('low', 'high')})


def run_once(auth_path, output, prior_cpu=0.):
    """Only invoked in the separately authorized, OS-accounted numerical worker."""
    software = runtime()
    software['public_seed_check'] = public_seed_check()
    auth, excluded = authorization(auth_path, output)
    output = Path(output).absolute()
    guard = Guard(output); guard(launch=True)
    claim(auth_path, output, auth)
    implementation = identity()
    atomic_json(output/'runtime.json', software)
    budget = Budget(output/'evaluations.jsonl', guard=guard, cpu_origin=-prior_cpu)
    records = [_empty(s) for s in panel()]
    previous = None
    try:
        seen = set(excluded)
        # Finish every coefficient-only selection and duplicate check before
        # constructing the first simulator or evaluating any QAOA objective.
        for row in records:
            budget.check_time()
            spec = row['spec']
            problem = getattr(problems, spec['generator'])(n=6, k=3, seed=spec['seeds']['qubo'], **spec['parameters'])
            digest = array_hash(problem.Q, problem.c)
            row.update(Q=problem.Q.tolist(), c=problem.c.tolist(), selection=select(problem))
            if digest in seen:
                raise ArithmeticError('Duplicate QUBO identity; no replacement permitted')
            seen.add(digest)
        atomic_json(output/'selection.json', records)
        previous = checkpoint(output, budget, 'selection')
        for wave in ('coarse', 'fine'):
            for row in records:
                for arm in row['spec']['arm_order']:
                    record = row['arms'][arm]
                    if not row['selection'] or not row['selection']['selected']:
                        record['unresolved_reason'] = 'NO_ALIGNMENT_CONTRAST'
                        continue
                    if budget.stop_reason or record.get('integrity_failure'):
                        record['unresolved_reason'] = budget.stop_reason or 'integrity_failure'
                        continue
                    if wave == 'fine' and record.get('unresolved_reason'):
                        continue
                    graph = row['selection']['selected'][arm]
                    landscape = f"{row['family']}:{row['replicate']}:{arm}"
                    path = output/f'{wave}-{landscape.replace(":", "-")}.json'
                    problem = problems.CardinalityQUBO(row['Q'], row['c'], 3)
                    sim = FeasibleSubspaceQAOA(problem, MixerSpec('c6', 6, tuple(graph['edges']), 'fixed relabeling'), initialization='uniform')
                    obj = Objective(sim, budget, landscape)
                    search = ReferenceSearch(obj, row['family'], row['replicate'])
                    record['bindings'] = bindings(problem, graph['edges'], landscape, implementation)
                    if obj.identity != record['bindings']['objective_identity']:
                        raise ArithmeticError('Simulator/selection identity disagrees')
                    if sim.feasible_span != row['selection']['span']:
                        raise ArithmeticError('Simulator/selection cost-span arithmetic disagrees')
                    record['mixer_width'] = float(np.ptp(sim._mixer_eigenvalues))
                    record['interior_points'] = interior(row['family'], row['replicate'], record['mixer_width']).tolist()
                    cache = output/f'cache-{landscape.replace(":", "-")}.npz'
                    try:
                        record['unresolved_reason'] = None
                        if wave == 'coarse':
                            _controls(record, pre_controls(obj, row['family'], row['replicate']))
                            _controls(record, transport_controls(obj))
                            result = search.coarse()
                            record['reference'] = result.record()
                            record['coarse_reference'] = result.record()
                            record['faces'] = search.face_records
                            if result.unresolved_reason and (budget.stop_reason or landscape in budget.arm_stops):
                                record['unresolved_reason'] = result.unresolved_reason
                            _cache_write(cache, obj)
                            record['cache_sha256'] = file_hash(cache)
                        else:
                            _cache_read(cache, record['cache_sha256'], obj)
                            result = search.fine(_result(record['reference']))
                            record['reference'] = result.record()
                            record['faces'].extend(search.face_records)
                            if budget.stop_reason or landscape in budget.arm_stops:
                                record['unresolved_reason'] = budget.stop_reason or budget.arm_stops[landscape]
                            elif all(any(c.route == route for c in result.candidates) for route in ('A', 'B')):
                                _controls(record, final_controls(obj, result))
                    except BudgetExhausted as error:
                        record['unresolved_reason'] = str(error)
                    except (ArithmeticError, ValueError) as error:
                        record['integrity_failure'] = True
                        record['unresolved_reason'] = str(error)
                        raise
                    finally:
                        atomic_json(path, record)
                        obj.release_cache()
            previous = checkpoint(output, budget, wave, previous=previous)
    except BudgetExhausted as error:
        for row in records:
            for a in row['arms'].values():
                if not a.get('reference'):
                    a['unresolved_reason'] = str(error)
    except (ArithmeticError, ValueError, OSError) as error:
        budget.stop_reason = 'integrity_failure'
        for row in records:
            for a in row['arms'].values():
                a['integrity_failure'] = True
                a['unresolved_reason'] = a['unresolved_reason'] or str(error)
    finally:
        finish(output, budget, records, implementation, file_hash(auth_path), previous=previous)
        budget.close()
