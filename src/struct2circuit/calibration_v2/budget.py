"""Cumulative v2 accounting, using the reviewed numerical/resource interface."""
from dataclasses import asdict, dataclass
from math import isfinite
from time import process_time

from ..calibration.budget import Budget as V1Budget, BudgetExhausted


@dataclass(frozen=True)
class Limits:
    development: int = 4_500_000
    total: int = 8_000_000
    validation_reserve: int = 3_500_000
    landscape: int = 120_000
    cpu_seconds: float = 28_800.

    def __post_init__(self):
        validate_limits(self)


# Explicit validation avoids constructing another dataclass inside __post_init__.
def validate_limits(limits):
    caps = dict(development=4_500_000, total=8_000_000, validation_reserve=3_500_000,
                landscape=120_000, cpu_seconds=28_800.)
    for name, value in asdict(limits).items():
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not isfinite(value) or not 0 < value <= caps[name]
            or name != 'cpu_seconds' and type(value) is not int):
            raise ValueError('Limits may be reduced for synthetic checks, never increased')
    if limits.development + limits.validation_reserve > limits.total:
        raise ValueError('Development would borrow protected validation evaluations')


class Budget(V1Budget):
    def __init__(self, limits=None, *, phase='development', cpu_origin=None, **clocks):
        super().__init__(**clocks)
        self.limits = limits or Limits()
        validate_limits(self.limits)
        if phase not in ('development', 'validation', 'diagnostics'):
            raise ValueError('Unknown v2 phase')
        self.phase = phase
        self.phases = {'development': 0, 'validation': 0, 'diagnostics': 0}
        if cpu_origin is not None:
            self.cpu_start = cpu_origin

    def request(self, landscape, domain):
        self.counts['objective_requests'] += 1
        self.domains[landscape, domain, 'requests'] += 1
        self.check_time()

    def charge(self, key, landscape, domain, account='essential', *, oracle=False, optional=False):
        self.check_time()
        if account not in ('essential', 'shared'):
            raise ValueError('Unknown provenance account')
        if optional and (self.phase != 'diagnostics' or not self.mandatory_complete):
            raise ValueError('Diagnostics require qualified validation')
        ceiling = ('total_evaluation_ceiling' if self.counts['actual_evaluations'] >= self.limits.total else
                   'landscape_evaluation_ceiling' if self.landscapes[landscape] >= self.limits.landscape else
                   'development_evaluation_ceiling' if self.phase == 'development' and
                   self.phases['development'] >= self.limits.development else None)
        if ceiling:
            self.stop_reason = ceiling
            raise BudgetExhausted(ceiling)
        self.counts['actual_evaluations'] += 1
        self.counts[account] += 1
        self.counts['optional'] += int(optional)
        self.counts['oracle_calls'] += int(oracle)
        self.phases[self.phase] += 1
        self.landscapes[landscape] += 1
        self.domains[landscape, domain, 'actual'] += 1
        self.unique.add(key)

    def snapshot(self):
        return super().snapshot() | dict(schema_version=2, phase=self.phase, phases=self.phases.copy(),
            cpu_measurement_cutoff=self.cpu_clock(), remaining=self.limits.total-self.counts['actual_evaluations'])

    @classmethod
    def from_snapshot(cls, s, *, phase, cpu_origin=None, **clocks):
        if s.get('schema_version') != 2:
            raise ValueError('A v1 budget cannot authorize v2')
        b = cls(Limits(**s['limits']), phase=phase, cpu_origin=cpu_origin, **clocks)
        b.counts.update(s['counters']); b.landscapes.update(s['per_landscape'])
        for row in s['per_domain']:
            b.domains[row['landscape'], row['domain'], row['counter']] = row['count']
        b.unique = {(i, bytes.fromhex(c)) for i, c in s['unique_keys']}
        b.phases = s['phases'].copy()
        b.prior_cpu, b.prior_wall = s['cpu_seconds'], s['wall_seconds']
        b.stop_reason = s['stop_reason']; b.mandatory_complete = s['mandatory_complete']
        actual = b.counts['actual_evaluations']
        values = [*b.counts.values(), *b.landscapes.values(), *b.domains.values(), *b.phases.values()]
        if (any(type(v) is not int or v < 0 for v in values)
            or set(b.phases) != {'development', 'validation', 'diagnostics'}
            or actual != sum(b.landscapes.values()) or actual != sum(b.phases.values())
            or actual != b.counts['essential']+b.counts['shared']
            or actual != sum(v for (_, _, k), v in b.domains.items() if k == 'actual')
            or actual > b.limits.total or b.phases['development'] > b.limits.development
            or b.counts['optional'] != b.phases['diagnostics']
            or any(v > b.limits.landscape for v in b.landscapes.values())
            or len(b.unique) != s['unique_parameter_points'] or len(b.unique) > actual
            or s['remaining'] != b.limits.total-actual
            or not all(isfinite(v) and v >= 0 for v in (b.prior_cpu,b.prior_wall))):
            raise ValueError('Invalid cumulative v2 accounting')
        if s.get('phase') not in b.phases:
            raise ValueError('Unknown persisted phase')
        if phase == 'validation' and s['phase'] == 'development' and (
            b.limits.total-actual < b.limits.validation_reserve):
            raise ValueError('Validation allocation is not protected')
        if phase == 'diagnostics' and (s['phase'] != 'validation' or not b.mandatory_complete):
            raise ValueError('Diagnostics require finalized qualified validation')
        b.check_time()
        return b
