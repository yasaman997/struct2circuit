#!/usr/bin/env python3
"""Run the bounded synthetic-only pre-freeze sensitivity study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from struct2circuit.sensitivity import SensitivityConfig, run_sensitivity  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--runtime-cap", type=float, default=60.0)
    parser.add_argument("--min-repetitions", type=int, default=200)
    parser.add_argument("--max-repetitions", type=int, default=1_000)
    parser.add_argument("--bootstrap-samples", type=int, default=399)
    parser.add_argument("--target-mcse", type=float, default=0.015)
    parser.add_argument("--random-baseline-sd", type=float, default=0.003)
    args = parser.parse_args()
    config = SensitivityConfig(
        seed=args.seed, runtime_cap_seconds=args.runtime_cap,
        min_repetitions=args.min_repetitions, max_repetitions=args.max_repetitions,
        bootstrap_samples=args.bootstrap_samples, target_mcse=args.target_mcse,
        random_baseline_sd=args.random_baseline_sd,
    )
    started = time.monotonic()
    result = run_sensitivity(config)
    elapsed = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {key: result[key] for key in ("status", "completed_grid_points", "planned_grid_points", "runtime_cap_reached")}
    summary["observed_runtime_seconds"] = elapsed
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
