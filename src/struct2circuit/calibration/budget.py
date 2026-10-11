"""Fail-closed counters for real computations, independent of solver counters."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import isfinite
from time import monotonic, process_time


class BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    essential: int = 6_000_000
    shared: int = 2_000_000
    total: int = 8_000_000
    landscape: int = 120_000
    optional: int = 250_000
    cpu_seconds: float = 28_800.

    def __post_init__(self):
        approved = (6_000_000, 2_000_000, 8_000_000, 120_000, 250_000, 28_800.)
        if any(not 0 < x <= cap for x, cap in zip(asdict(self).values(), approved)):
            raise ValueError("Limits may be reduced for toys, never increased")
        if any(type(x) is not int for x in list(asdict(self).values())[:5]):
            raise ValueError("Evaluation limits must be integers")


class Budget:
    def __init__(self, limits=None, *, cpu_clock=process_time, wall_clock=monotonic):
        self.limits = limits or Limits()
        self.cpu_clock, self.wall_clock = cpu_clock, wall_clock
        self.cpu_start, self.wall_start = cpu_clock(), wall_clock()
        self.prior_cpu = self.prior_wall = 0.
        self.counts = Counter()
        self.landscapes, self.domains = Counter(), Counter()
        self.unique = set()
        self.stop_reason = None
        self.mandatory_complete = False
        self.resource_guard = None

    def check_time(self):
        if self.stop_reason:
            raise BudgetExhausted(self.stop_reason)
        if self.prior_cpu+self.cpu_clock()-self.cpu_start >= self.limits.cpu_seconds:
            self.stop_reason = "cpu_ceiling"
            raise BudgetExhausted(self.stop_reason)
        if self.resource_guard is not None:
            self.resource_guard.check(self)

    def request(self, landscape, domain):
        self.counts['objective_requests'] += 1
        self.domains[(landscape, domain, 'requests')] += 1
        self.check_time()
        for name, used, ceiling in (('total_evaluation_ceiling', self.counts['actual_evaluations'], self.limits.total),
                                   ('landscape_evaluation_ceiling', self.landscapes[landscape], self.limits.landscape),
                                   ('shared_evaluation_ceiling', self.counts['shared'], self.limits.shared),
                                   ('optional_evaluation_ceiling', self.counts['optional'], self.limits.optional)):
            if used >= ceiling:
                self.stop_reason = name
                raise BudgetExhausted(self.stop_reason)

    def charge(self, key, landscape, domain, account="essential", *, oracle=False, optional=False):
        self.check_time()
        if account not in ("essential", "shared"):
            raise ValueError("Unknown budget account")
        if optional and (not self.mandatory_complete or account != "shared"):
            raise ValueError("Optional work requires completed mandatory work and shared reserve")
        charge_account = account
        if account == 'essential' and self.counts['essential'] >= self.limits.essential:
            charge_account = 'shared'
        exhausted = (self.counts['actual_evaluations'] >= self.limits.total
                     or self.landscapes[landscape] >= self.limits.landscape
                     or self.counts[charge_account] >= getattr(self.limits, charge_account)
                     or (optional and self.counts['optional'] >= self.limits.optional))
        if exhausted:
            self.stop_reason = "evaluation_ceiling"
            raise BudgetExhausted(self.stop_reason)
        self.counts['actual_evaluations'] += 1
        self.counts[charge_account] += 1
        self.counts['optional'] += int(optional)
        self.counts['oracle_calls'] += int(oracle)
        self.counts['essential_to_shared'] += int(account != charge_account)
        self.landscapes[landscape] += 1
        self.domains[(landscape, domain, 'actual')] += 1
        self.unique.add(key)

    def snapshot(self):
        return dict(limits=asdict(self.limits), counters=dict(self.counts), unique_parameter_points=len(self.unique),
                    per_landscape=dict(self.landscapes),
                    per_domain=[dict(landscape=k[0], domain=k[1], counter=k[2], count=v) for k, v in self.domains.items()],
                    cpu_seconds=self.prior_cpu+self.cpu_clock()-self.cpu_start, wall_seconds=self.prior_wall+self.wall_clock()-self.wall_start,
                    stop_reason=self.stop_reason, mandatory_complete=self.mandatory_complete,
                    unique_keys=[(identity, coordinates.hex()) for identity, coordinates in sorted(self.unique)])

    @classmethod
    def from_snapshot(cls, snapshot):
        budget = cls(Limits(**snapshot['limits']))
        budget.counts.update(snapshot['counters'])
        budget.landscapes.update(snapshot['per_landscape'])
        for row in snapshot['per_domain']:
            budget.domains[(row['landscape'], row['domain'], row['counter'])] = row['count']
        identities = {}
        budget.unique = {(identities.setdefault(identity, identity), bytes.fromhex(coordinates))
                         for identity, coordinates in snapshot['unique_keys']}
        budget.prior_cpu = snapshot['cpu_seconds']
        budget.prior_wall = snapshot['wall_seconds']
        if (any(type(v) is not int or v < 0 for v in (*budget.counts.values(), *budget.landscapes.values(), *budget.domains.values()))
            or not all(isfinite(v) and v >= 0 for v in (budget.prior_cpu, budget.prior_wall))
            or budget.counts['actual_evaluations'] != budget.counts['essential']+budget.counts['shared']
            or budget.counts['actual_evaluations'] != sum(budget.landscapes.values())
            or budget.counts['essential'] > budget.limits.essential
            or budget.counts['shared'] > budget.limits.shared
            or budget.counts['actual_evaluations'] > budget.limits.total
            or any(v > budget.limits.landscape for v in budget.landscapes.values())
            or len(budget.unique) != snapshot['unique_parameter_points']
            or len(budget.unique) > budget.counts['actual_evaluations']
            or budget.counts['optional'] > min(budget.limits.optional, budget.counts['shared'])
            or sum(v for (_, _, kind), v in budget.domains.items() if kind == 'actual') != budget.counts['actual_evaluations']):
            raise ValueError('Invalid persisted budget accounting')
        budget.stop_reason = snapshot['stop_reason']
        budget.check_time()
        return budget
