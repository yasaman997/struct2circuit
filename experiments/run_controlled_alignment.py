#!/usr/bin/env python3
"""Default is metadata only. Numerical execution requires a separate capability."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import process_time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--verify', type=Path, help='Read-only stored scalar/evidence verification')
    p.add_argument('--authorization', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--_coordinator', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.verify:
        from struct2circuit.alignment_experiment.storage import verify
        print(json.dumps(verify(a.verify), indent=2, allow_nan=False)); return
    if not a.execute:
        from struct2circuit.alignment_experiment.config import identity, manifest
        print(json.dumps(dict(manifest=manifest(), implementation=identity()), indent=2, allow_nan=False)); return
    if not a.authorization or not a.output:
        p.error('Execution requires separate authorization and a fresh output')
    if not a._coordinator:
        child = subprocess.Popen([sys.executable, __file__, *sys.argv[1:], '--_coordinator'])
        _, status, usage = os.wait4(child.pid, 0)
        if os.waitstatus_to_exitcode(status):
            raise SystemExit('Coordinator failed; preserve partial evidence; no restart')
        from struct2circuit.alignment_experiment.storage import settle
        settle(a.output.absolute(), usage.ru_utime+usage.ru_stime)
        print('AWAITING_INDEPENDENT_VERIFICATION')
        return
    from struct2circuit.alignment_experiment.config import runtime
    from struct2circuit.alignment_experiment.storage import authorization
    from struct2circuit.calibration_v2.storage import atomic_json
    runtime(); authorization(a.authorization, a.output)  # reject before spawning
    code = ('import sys; from struct2circuit.alignment_experiment.driver import run_once; '
            'run_once(sys.argv[1],sys.argv[2],float(sys.argv[3]))')
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    child = subprocess.Popen([sys.executable, '-c', code, str(a.authorization), str(a.output), str(process_time())], env=env)
    _, status, usage = os.wait4(child.pid, 0)
    exit_code = os.waitstatus_to_exitcode(status)
    if exit_code:
        raise SystemExit(f'Worker failed ({exit_code}); authorization cannot be reused; preserve partial evidence')
    atomic_json(a.output/'worker-ended.json', dict(cpu_seconds=usage.ru_utime+usage.ru_stime+process_time(), process_ended=True))


if __name__ == '__main__':
    main()
