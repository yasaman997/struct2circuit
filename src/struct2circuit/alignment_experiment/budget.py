"""One-shot, control-inclusive resource ledger. Existing journals never restart."""
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from time import monotonic, process_time
import json
import os

from ..calibration.budget import BudgetExhausted
from ..calibration.resources import GIB, resource_sample
from ..calibration.storage import canonical_hash


class ArmExhausted(BudgetExhausted):
    pass


class Budget:
    def __init__(self, journal, *, total=1280000, arm=20000, numerical_cpu=3300., cpu=3600.,
                 cpu_clock=process_time, wall_clock=monotonic, cpu_origin=0., guard=None):
        if not (0 < total <= 1280000 and 0 < arm <= 20000 and 0 < numerical_cpu <= 3300 and numerical_cpu < cpu <= 3600):
            raise ValueError('Synthetic reductions only; approved ceilings cannot increase')
        self.path = Path(journal)
        self.file = self.path.open('x', encoding='utf8')  # no resume or reset of spent capacity
        self.total, self.arm, self.numerical_cpu, self.cpu_limit = total, arm, numerical_cpu, cpu
        self.cpu_clock, self.wall_clock = cpu_clock, wall_clock
        self.cpu_origin, self.wall_origin = cpu_origin, wall_clock()
        self.guard = guard
        self.counts, self.landscapes, self.domains = Counter(), Counter(), Counter()
        self.unique, self.stop_reason, self.arm_stops = set(), None, {}
        self.previous = None
        self.results_path = self.path.with_suffix('.results.jsonl')
        self.results_file = self.results_path.open('x', encoding='utf8')
        self.result_tail, self.result_count = None, 0
        self.traces = []

    def record(self, payload):
        payload = dict(payload, sequence=self.result_count+1, previous=self.result_tail)
        digest = canonical_hash(payload)
        self.results_file.write(json.dumps(dict(payload=payload, sha256=digest), sort_keys=True, allow_nan=False)+'\n')
        self.results_file.flush()
        self.result_tail, self.result_count = digest, self.result_count+1
        return digest

    @contextmanager
    def trace(self):
        work = dict(requests=[], computations=[])
        self.traces.append(work)
        try:
            yield work
        finally:
            assert self.traces.pop() is work

    def outcome(self, computation, value=None, *, error=None, gradient_started=False, matrix_exponentials=0):
        return self.record(dict(kind='result', computation_id=computation,
            status='failed' if error is not None else 'success', value=value,
            gradient_started=gradient_started, matrix_exponentials=matrix_exponentials,
            error=None if error is None else dict(type=type(error).__name__, reason=str(error))))

    def resolve(self, request, computation, *, cached=False, error=None):
        if cached:
            self.counts['cache_hits'] += 1
        self.record(dict(kind='resolution', request_id=request, computation_id=computation,
            cached=cached, error=None if error is None else dict(type=type(error).__name__, reason=str(error))))

    def check_time(self):
        if self.stop_reason:
            raise BudgetExhausted(self.stop_reason)
        if self.cpu_clock()-self.cpu_origin >= self.numerical_cpu:
            self.stop_reason = 'numerical_cpu_cutoff'
        elif self.wall_clock()-self.wall_origin >= 7200:
            self.stop_reason = 'wall_ceiling'
        if self.guard and not self.stop_reason:
            self.guard(self)
        if self.stop_reason:
            raise BudgetExhausted(self.stop_reason)

    def request(self, landscape, domain, *, key=None, fresh=False, category='native'):
        if domain != 'B00':
            raise ValueError('Only B00 numerical work is permitted')
        self.check_time()
        self.counts['objective_requests'] += 1
        request = self.record(dict(kind='request', landscape=landscape, domain=domain,
            identity=key[0] if key else None, coordinates=key[1].hex() if key else None,
            fresh=fresh, category=category))
        for trace in self.traces:
            trace['requests'].append(request)
        return request

    def charge(self, key, landscape, domain, account='essential', *, oracle=False, optional=False, derivative=False,
               request_id=None, fresh=False):
        self.check_time()
        if domain != 'B00' or optional or account != 'essential':
            raise ValueError('No broader-domain or optional execution')
        if self.counts['actual_evaluations'] >= self.total:
            self.stop_reason = 'total_evaluation_ceiling'
            raise BudgetExhausted(self.stop_reason)
        if self.landscapes[landscape] >= self.arm:
            self.arm_stops[landscape] = 'arm_evaluation_ceiling'
            raise ArmExhausted('arm_evaluation_ceiling')
        event = dict(sequence=self.counts['actual_evaluations']+1, landscape=landscape, domain=domain,
                     identity=key[0], coordinates=key[1].hex(), oracle=oracle, derivative=derivative,
                     request_id=request_id, fresh=fresh,
                     previous=self.previous)
        digest = canonical_hash(event)
        self.file.write(json.dumps(dict(payload=event, sha256=digest), sort_keys=True, allow_nan=False)+'\n')
        self.file.flush()  # record an admitted computation even when its backend fails
        self.previous = digest
        self.counts['actual_evaluations'] += 1
        self.counts['essential'] += 1
        self.counts['oracle_calls'] += int(oracle)
        self.counts['derivative_evolutions'] += int(derivative)
        self.landscapes[landscape] += 1
        self.domains[(landscape, domain)] += 1
        self.unique.add(key)
        for trace in self.traces:
            trace['computations'].append(digest)
        return digest

    def snapshot(self):
        self.file.flush(); os.fsync(self.file.fileno())
        self.results_file.flush(); os.fsync(self.results_file.fileno())
        return dict(counters=dict(self.counts), per_arm=dict(self.landscapes),
                    per_domain=[dict(arm=a, domain=d, count=v) for (a, d), v in sorted(self.domains.items())],
                    journal_tail=self.previous, result_tail=self.result_tail, result_count=self.result_count,
                    cpu_seconds=self.cpu_clock()-self.cpu_origin,
                    wall_seconds=self.wall_clock()-self.wall_origin, stop_reason=self.stop_reason,
                    arm_stops=self.arm_stops, limits=dict(total=self.total, arm=self.arm,
                    numerical_cpu=self.numerical_cpu, cpu=self.cpu_limit))

    def close(self):
        self.file.flush(); os.fsync(self.file.fileno()); self.file.close()
        self.results_file.flush(); os.fsync(self.results_file.fileno()); self.results_file.close()


class Guard:
    def __init__(self, output, sample=resource_sample, clock=monotonic):
        self.output, self.sample, self.clock, self.last = output, sample, clock, float('-inf')

    def __call__(self, budget=None, *, launch=False):
        now = self.clock()
        if not launch and now-self.last < .5:
            return
        self.last = now
        try:
            total, available, rss, free = self.sample((self.output,))
            if self.sample is resource_sample:
                # Official runner has one numerical worker and its coordinator;
                # account for the coordinator's resident pages as well.
                rss += int(Path(f'/proc/{os.getppid()}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
            if rss >= 4*GIB or available < 256*1024**2:
                reason = 'memory_ceiling'
            elif free < (5*GIB if launch else GIB):
                reason = 'disk_headroom'
            else:
                return
        except (OSError, ValueError, KeyError):
            reason = 'resource_information_unavailable'
        if budget:
            budget.stop_reason = reason
        raise BudgetExhausted(reason)
