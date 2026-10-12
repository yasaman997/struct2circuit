"""Operational headroom checks, separate from scientific acceptance thresholds.

Linux memory/cgroup information is required. These sampled checks cannot prevent
every native allocation spike or competing process from exhausting resources.
"""
from pathlib import Path
import os
import shutil
from time import monotonic

from .budget import BudgetExhausted

GIB = 1024**3
MIB = 1024**2


class ResourceExhausted(BudgetExhausted):
    """Unfinished operational work, never a scientific numerical failure."""


def resource_sample(paths):
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, _, value = line.partition(':')
        if name in ('MemTotal', 'MemAvailable'):
            memory[name] = int(value.split()[0])*1024
    total, available = memory['MemTotal'], memory['MemAvailable']
    cgroup = Path('/sys/fs/cgroup')
    if (cgroup/'memory.max').is_file():
        cap = (cgroup/'memory.max').read_text().strip()
        if cap != 'max':
            cap = int(cap)
            total = min(total, cap)
            available = min(available, cap-int((cgroup/'memory.current').read_text()))
    elif (cgroup/'memory/memory.limit_in_bytes').is_file():
        cap = int((cgroup/'memory/memory.limit_in_bytes').read_text())
        total = min(total, cap)
        available = min(available, cap-int((cgroup/'memory/memory.usage_in_bytes').read_text()))
    else:
        raise OSError('Readable cgroup memory limits are required')
    rss = int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    free = []
    for path in paths:
        parent = Path(path).absolute()
        while not parent.exists(): parent = parent.parent
        free.append(shutil.disk_usage(parent).free)
    if min(total, available, rss, *free) <= 0:
        raise OSError('Invalid or exhausted memory/disk information')
    return total, available, rss, min(free)


class ResourceGuard:
    def __init__(self, output, lock_output=None, *, sample=resource_sample, clock=monotonic):
        self.paths = (output,) if lock_output is None else (output, lock_output)
        self.sample, self.clock = sample, clock
        self.last_time, self.last_requests = float('-inf'), -256

    def check(self, budget=None, *, force=False, extra_memory=0):
        now = self.clock()
        requests = budget.counts['objective_requests'] if budget else 0
        if not force and now-self.last_time < 1. and requests-self.last_requests < 256:
            return
        self.last_time, self.last_requests = now, requests
        count = len(budget.unique) if budget else 0
        # Stage 6A measured up to ~728 extra bytes/key in lock verification.
        # Reserve 1,100 bytes/key plus 512 MiB for finalization/native work.
        # Disk reserve covers budget, lock and manifest copies, beyond live rows.
        memory_reserve = 512*MIB+1100*count+extra_memory
        disk_reserve = 512*MIB+512*count
        try:
            total, available, rss, free = self.sample(self.paths)
            if min(total, available, rss, free) <= 0:
                raise OSError('Invalid resource sample')
        except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
            reason = 'resource_information_unavailable'
            detail = f'Required Linux memory/disk information unavailable: {error}'
        else:
            allowance = min(24*GIB, int(.75*total))
            if rss+memory_reserve > allowance or available < memory_reserve:
                reason = 'resource_memory_headroom'
                detail = (f'Insufficient memory headroom: RSS {rss} bytes, reserve {memory_reserve} bytes, '
                          f'available {available} bytes, operational allowance {allowance} bytes')
            elif free < disk_reserve:
                reason = 'resource_disk_headroom'
                detail = f'Insufficient output disk headroom: free {free} bytes, reserve {disk_reserve} bytes'
            else:
                return
        if budget is not None and not budget.stop_reason:
            budget.stop_reason = reason
        raise ResourceExhausted(detail+'; required calibration work is INCONCLUSIVE')
