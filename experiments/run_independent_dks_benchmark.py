#!/usr/bin/env python3
"""Run the first independent Struct2Circuit benchmark on weighted densest-k-subgraph instances.

This benchmark is intentionally exploratory. It uses a fresh seed namespace and a
problem family that was not used in the original 24-instance block-correlated pilot.
It must not be treated as the frozen benchmark-v1 evaluation.
The historical strong-|Q| heuristic failed to generalize on this family. Legacy
gamma units preserve that experiment; undefined gaps stop aggregation after raw
rows have been saved, with no silent omission of instances or random controls.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from struct2circuit.mixers import MixerSpec, graph_connected, random_connected_mixer, ring_mixer, structure_conditioned_mixer
from struct2circuit.analysis import require_defined_gaps
from struct2circuit.optimize import optimize_p1
from struct2circuit.problems import weighted_densest_k_subgraph_qubo
from struct2circuit.simulator import FeasibleSubspaceQAOA


def inverse_structure_mixer(q: np.ndarray, edge_budget: int) -> MixerSpec:
    """Use the weakest |Q_ij| interactions first, while enforcing connectivity.

    This is a mechanism control, not a proposed algorithm.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("q must be square")
    n = q.shape[0]
    maximum = n * (n - 1) // 2
    if not n - 1 <= edge_budget <= maximum:
        raise ValueError(f"edge_budget must lie in [{n - 1}, {maximum}]")

    ranked = sorted(
        ((abs(float(q[i, j])), i, j) for i, j in combinations(range(n), 2)),
        key=lambda item: (item[0], item[1], item[2]),
    )

    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    chosen: list[tuple[int, int]] = []
    chosen_set: set[tuple[int, int]] = set()

    for _, i, j in ranked:
        if union(i, j):
            chosen.append((i, j))
            chosen_set.add((i, j))
            if len(chosen) == n - 1:
                break

    for _, i, j in ranked:
        if len(chosen) >= edge_budget:
            break
        edge = (i, j)
        if edge not in chosen_set:
            chosen.append(edge)
            chosen_set.add(edge)

    mixer = MixerSpec(
        "inverse_structure",
        n,
        tuple(chosen),
        "minimum-|Q_ij| spanning tree plus weakest remaining interactions",
    )
    if not graph_connected(n, mixer.edges):
        raise RuntimeError("inverse structure mixer is not connected")
    return mixer


def interaction_alignment(q: np.ndarray, edges: tuple[tuple[int, int], ...]) -> float:
    """Ratio of selected |Q_ij| mass to the maximum mass achievable at this edge budget."""
    weights = np.asarray(
        [abs(float(q[i, j])) for i, j in combinations(range(q.shape[0]), 2)],
        dtype=float,
    )
    selected = np.asarray([abs(float(q[i, j])) for i, j in edges], dtype=float)
    top = np.sort(weights)[-len(edges):]
    denominator = float(np.sum(top))
    return 1.0 if denominator <= 1e-15 else float(np.sum(selected) / denominator)


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
    parser.add_argument("--seed-base", type=int, default=900_000_000)
    parser.add_argument("--random-seed-base", type=int, default=1_700_000_000)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.instances < 4:
        raise SystemExit("--instances must be at least 4")
    if args.edge_budget < args.n - 1:
        raise SystemExit("--edge-budget must be at least n-1")
    if args.random_replicates < 1:
        raise SystemExit("--random-replicates must be at least 1 for the random-mean comparator")
    args.output.mkdir(parents=True, exist_ok=True)

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

        mixers: list[tuple[str, MixerSpec]] = [
            ("ring", ring_mixer(args.n)),
            ("structure", structure_conditioned_mixer(problem.Q, args.edge_budget)),
            ("inverse_structure", inverse_structure_mixer(problem.Q, args.edge_budget)),
        ]

        for replicate in range(args.random_replicates):
            seed = args.random_seed_base + 997 * instance + replicate
            mixers.append(
                (
                    f"random_{replicate:02d}",
                    random_connected_mixer(args.n, args.edge_budget, seed),
                )
            )

        for method, mixer in mixers:
            simulator = FeasibleSubspaceQAOA(problem, mixer)
            result = optimize_p1(simulator, grid_size=args.grid_size, gamma_scale="legacy")
            rows.append(
                {
                    "instance": instance,
                    "problem_seed": problem_seed,
                    "method": method,
                    "mixer_edges": mixer.edge_count,
                    "normalized_gap": result.result.normalized_gap,
                    "cost_status": result.result.cost_status,
                    "probability_optimum": result.result.probability_optimum,
                    "gamma": result.gamma[0],
                    "beta": result.beta[0],
                    "gamma_scale": "legacy",
                    "finite_objective": result.finite_objective,
                    "selected_source": result.selected_source,
                    "selected_source_success": result.selected_source_success,
                    "lbfgsb_success": result.lbfgsb_success,
                    "powell_success": result.powell_success,
                    "local_refinement_converged": result.local_refinement_converged,
                    "objective_evaluations": result.objective_evaluations,
                    "interaction_alignment": interaction_alignment(problem.Q, mixer.edges),
                }
            )

    raw = pd.DataFrame(rows)
    raw_path = args.output / "independent_dks_benchmark_v1_raw.csv"
    raw.to_csv(raw_path, index=False, na_rep="")

    require_defined_gaps(
        raw, "method", expected_instances=range(args.instances), required_methods=(
            "ring", "structure", "inverse_structure",
            *(f"random_{replicate:02d}" for replicate in range(args.random_replicates)),
        ),
    )
    pivot = raw.pivot(index="instance", columns="method", values="normalized_gap")
    random_columns = [c for c in pivot.columns if c.startswith("random_")]
    random_mean = pivot[random_columns].mean(axis=1)

    instance_summary = pd.DataFrame(
        {
            "instance": pivot.index,
            "problem_seed": [
                int(raw[(raw["instance"] == i) & (raw["method"] == "ring")]["problem_seed"].iloc[0])
                for i in pivot.index
            ],
            "structure_gap": pivot["structure"],
            "ring_gap": pivot["ring"],
            "random_mean_gap": random_mean,
            "inverse_structure_gap": pivot["inverse_structure"],
            "structure_minus_ring": pivot["ring"] - pivot["structure"],
            "structure_minus_random_mean": random_mean - pivot["structure"],
            "structure_minus_inverse": pivot["inverse_structure"] - pivot["structure"],
            "structure_alignment": raw[raw["method"] == "structure"].set_index("instance")["interaction_alignment"],
            "ring_alignment": raw[raw["method"] == "ring"].set_index("instance")["interaction_alignment"],
            "inverse_alignment": raw[raw["method"] == "inverse_structure"].set_index("instance")["interaction_alignment"],
        }
    )
    summary_path = args.output / "independent_dks_benchmark_v1_instances.csv"
    instance_summary.to_csv(summary_path, float_format="%.8f")

    print(f"wrote raw results to {raw_path}")
    print(f"wrote instance summary to {summary_path}")


if __name__ == "__main__":
    main()
