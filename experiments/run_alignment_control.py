#!/usr/bin/env python3
"""Measure sensitivity to initialization for the same mixer graphs.

This diagnostic does not isolate a pure topology effect: each spectral reference
is part of a topology+initialization algorithmic configuration. Both mixer
spectral extrema are reported so the interpretation is explicit under the
chosen Hamiltonian sign convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from struct2circuit.mixers import random_connected_mixer, ring_mixer, structure_conditioned_mixer
from struct2circuit.optimize import optimize_p1
from struct2circuit.problems import weighted_densest_k_subgraph_qubo
from struct2circuit.simulator import FeasibleSubspaceQAOA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=24)
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--density", type=float, default=0.55)
    parser.add_argument("--planted-strength", type=float, default=1.8)
    parser.add_argument("--edge-budget", type=int, default=9)
    parser.add_argument("--random-replicates", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--seed-base", type=int, default=910_000_000)
    parser.add_argument("--random-seed-base", type=int, default=1_710_000_000)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "alignment_control.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.edge_budget < args.n - 1:
        raise SystemExit("--edge-budget must be at least n-1")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for instance in range(args.instances):
        problem_seed = args.seed_base + 10007 * instance
        problem = weighted_densest_k_subgraph_qubo(
            args.n,
            args.k,
            density=args.density,
            seed=problem_seed,
            planted_community_strength=args.planted_strength,
        )
        mixers = [
            ("ring", ring_mixer(args.n)),
            ("structure", structure_conditioned_mixer(problem.Q, args.edge_budget)),
        ]
        for replicate in range(args.random_replicates):
            seed = args.random_seed_base + 997 * instance + replicate
            mixers.append((f"random_{replicate:02d}", random_connected_mixer(args.n, args.edge_budget, seed)))

        for method, mixer in mixers:
            for initialization in ("uniform", "mixer_low", "mixer_high"):
                simulator = FeasibleSubspaceQAOA(
                    problem,
                    mixer,
                    initialization=initialization,
                )
                result = optimize_p1(simulator, grid_size=args.grid_size)
                rows.append(
                    {
                        "instance": instance,
                        "problem_seed": problem_seed,
                        "method": method,
                        "initialization": initialization,
                        "uniform_low_eigenspace_fidelity": simulator.uniform_mixer_extremal_fidelity("low"),
                        "uniform_high_eigenspace_fidelity": simulator.uniform_mixer_extremal_fidelity("high"),
                        "normalized_gap": result.result.normalized_gap,
                        "probability_optimum": result.result.probability_optimum,
                        "gamma": result.gamma[0],
                        "beta": result.beta[0],
                        "finite_objective": result.finite_objective,
                        "local_refinement_converged": result.local_refinement_converged,
                    }
                )

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote alignment-control results to {args.output}")


if __name__ == "__main__":
    main()
