#!/usr/bin/env python3
"""Run the deterministic Struct2Circuit depth-one pilot."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import pandas as pd  # noqa: E402

from struct2circuit.analysis import plot_pilot, save_summary, summarize_pilot, write_report  # noqa: E402
from struct2circuit.mixers import complete_mixer, ring_mixer, structure_conditioned_mixer  # noqa: E402
from struct2circuit.optimize import optimize_p1  # noqa: E402
from struct2circuit.problems import block_correlated_qubo  # noqa: E402
from struct2circuit.simulator import FeasibleSubspaceQAOA  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=24)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--grid-size", type=int, default=17)
    parser.add_argument("--edge-budget", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.instances < 4:
        raise SystemExit("--instances must be at least 4")
    edge_budget = args.n if args.edge_budget is None else args.edge_budget
    args.output.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for instance in range(args.instances):
        problem_seed = args.seed + 104729 * instance
        block_strength = 0.58 + 0.30 * (instance / max(1, args.instances - 1))
        problem = block_correlated_qubo(
            args.n,
            args.k,
            problem_seed,
            block_strength=block_strength,
        )
        mixers = [
            ring_mixer(args.n),
            structure_conditioned_mixer(problem.Q, edge_budget),
            complete_mixer(args.n),
        ]
        for mixer in mixers:
            simulator = FeasibleSubspaceQAOA(problem, mixer)
            optimum = optimize_p1(simulator, grid_size=args.grid_size)
            result = optimum.result
            records.append({
                "instance": instance,
                "problem_seed": problem_seed,
                "problem_name": problem.name,
                "n": args.n,
                "k": args.k,
                "feasible_dimension": simulator.feasible_dimension,
                "block_strength": block_strength,
                "mixer": mixer.name,
                "mixer_edges": mixer.edge_count,
                "gamma": optimum.gamma[0],
                "beta": optimum.beta[0],
                "expectation": result.expectation,
                "normalized_gap": result.normalized_gap,
                "probability_optimum": result.probability_optimum,
                "feasibility_probability": result.feasibility_probability,
                "state_norm": result.state_norm,
                "objective_evaluations": optimum.objective_evaluations,
                "optimizer_success": optimum.optimizer_success,
                "local_refinement_converged": optimum.local_refinement_converged,
            })

    results = pd.DataFrame.from_records(records)
    config = {
        "instances": args.instances,
        "n": args.n,
        "k": args.k,
        "p": 1,
        "seed": args.seed,
        "grid_size": args.grid_size,
        "equal_budget_mixer_edges": edge_budget,
        "generator": "block_correlated_qubo",
    }
    summary = summarize_pilot(results, config)
    results.to_csv(args.output / "pilot_results.csv", index=False)
    save_summary(summary, args.output / "pilot_summary.json")
    write_report(summary, args.output / "pilot_report.md")
    plot_pilot(results, args.output / "pilot_quality_resource.png")
    pair = summary["paired_structure_vs_ring"]
    print(f"wrote {len(results)} rows to {args.output}")
    print(
        "structure vs ring: median gap reduction="
        f"{pair['median_gap_reduction']:.6f}, "
        f"wins/ties/losses={pair['wins']}/{pair['ties']}/{pair['losses']}, "
        f"one-sided p={pair['one_sided_wilcoxon_p']:.6g}"
    )


if __name__ == "__main__":
    main()
