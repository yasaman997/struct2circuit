"""Manifest preview by default; official execution needs separate authorization."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Set numerical threads before importing NumPy/SciPy in this process.
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from struct2circuit.calibration.config import build_manifest, verify_protocol
from struct2circuit.calibration.driver import run_official_phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('development', 'validation'))
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validation-lock', type=Path)
    parser.add_argument('--write-validation-lock', type=Path)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).resolve().parents[1]/'paper/numerical_calibration_protocol_v1.md')
    args = parser.parse_args()
    verify_protocol(args.protocol)
    if args.phase is None:
        print(json.dumps(build_manifest(), indent=2))
        return
    if args.authorization is None or args.output is None:
        parser.error('Official execution requires --authorization and a fresh --output; manifest preview is the default')
    print(run_official_phase(args.output, phase=args.phase, authorization=args.authorization,
                             protocol=args.protocol, lock=args.validation_lock, lock_output=args.write_validation_lock))


if __name__ == '__main__': main()
