"""Mixer graphs and exact Hamming-weight-preserving XY Hamiltonians."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray


Edge = tuple[int, int]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int8]


@dataclass(frozen=True)
class MixerSpec:
    name: str
    n: int
    edges: tuple[Edge, ...]
    construction: str

    def __post_init__(self) -> None:
        normalized = tuple(sorted((min(i, j), max(i, j)) for i, j in self.edges))
        if len(set(normalized)) != len(normalized):
            raise ValueError("mixer edges must be unique")
        if any(i == j or i < 0 or j >= self.n for i, j in normalized):
            raise ValueError("invalid mixer edge")
        object.__setattr__(self, "edges", normalized)
        if not graph_connected(self.n, normalized):
            raise ValueError("mixer graph must be connected")

    @property
    def edge_count(self) -> int:
        return len(self.edges)


def graph_connected(n: int, edges: tuple[Edge, ...] | list[Edge]) -> bool:
    adjacency = [set() for _ in range(n)]
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    seen = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node] - seen:
            seen.add(neighbor)
            frontier.append(neighbor)
    return len(seen) == n


def ring_mixer(n: int) -> MixerSpec:
    edges = tuple(sorted({(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)}))
    return MixerSpec("ring", n, edges, "fixed nearest-neighbor cycle")


def complete_mixer(n: int) -> MixerSpec:
    return MixerSpec("complete", n, tuple(combinations(range(n), 2)), "all-to-all reference")


def random_connected_mixer(
    n: int,
    edge_budget: int,
    seed: int,
    *,
    max_attempts: int = 10_000,
) -> MixerSpec:
    """Uniformly sample a fixed-size edge set, conditional on connectivity.

    Rejection sampling is simple and unbiased for pilot-scale graphs, but can
    become inefficient when connectivity is rare at larger sizes.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer of at least 2")
    if not isinstance(edge_budget, int) or isinstance(edge_budget, bool):
        raise ValueError("edge_budget must be an integer")
    maximum = n * (n - 1) // 2
    if not n - 1 <= edge_budget <= maximum:
        raise ValueError(f"edge_budget must lie in [{n - 1}, {maximum}]")
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")

    possible_edges = tuple(combinations(range(n), 2))
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        indices = rng.choice(len(possible_edges), size=edge_budget, replace=False)
        edges = tuple(sorted(possible_edges[int(index)] for index in indices))
        if graph_connected(n, edges):
            return MixerSpec(
                "random_connected",
                n,
                edges,
                f"uniform fixed-edge rejection sampling; seed={seed}; max_attempts={max_attempts}",
            )
    raise RuntimeError(
        f"could not sample a connected mixer in {max_attempts} attempts "
        f"(n={n}, edge_budget={edge_budget}, seed={seed})"
    )


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def structure_conditioned_mixer(Q: FloatArray, edge_budget: int) -> MixerSpec:
    """Select strong QUBO interactions while guaranteeing connectivity.

    A maximum-weight spanning tree under ``abs(Q_ij)`` is built first. The
    strongest unused edges are then added until the requested budget is met.
    This is the first transparent, non-learned structure-conditioned baseline;
    later work will learn the scoring rule under the same verifier.
    """
    q = np.asarray(Q, dtype=float)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("Q must be square")
    n = q.shape[0]
    maximum = n * (n - 1) // 2
    if not n - 1 <= edge_budget <= maximum:
        raise ValueError(f"edge_budget must lie in [{n - 1}, {maximum}]")

    ranked = sorted(
        ((-abs(float(q[i, j])), i, j) for i, j in combinations(range(n), 2)),
        key=lambda item: (item[0], item[1], item[2]),
    )
    uf = _UnionFind(n)
    chosen: list[Edge] = []
    chosen_set: set[Edge] = set()
    for _, i, j in ranked:
        if uf.union(i, j):
            chosen.append((i, j))
            chosen_set.add((i, j))
            if len(chosen) == n - 1:
                break
    for _, i, j in ranked:
        if len(chosen) >= edge_budget:
            break
        if (i, j) not in chosen_set:
            chosen.append((i, j))
            chosen_set.add((i, j))
    return MixerSpec(
        "structure",
        n,
        tuple(chosen),
        "maximum-|Q_ij| spanning tree plus strongest remaining interactions",
    )


def xy_mixer_hamiltonian(basis: IntArray, mixer: MixerSpec) -> FloatArray:
    """Construct ``sum_(i,j in E) (X_i X_j + Y_i Y_j)/2`` in a fixed-weight basis."""
    states = np.asarray(basis, dtype=np.int8)
    if states.ndim != 2 or states.shape[1] != mixer.n:
        raise ValueError("basis shape does not match mixer")
    index = {tuple(int(v) for v in row): idx for idx, row in enumerate(states)}
    h = np.zeros((len(states), len(states)), dtype=float)
    for a, state in enumerate(states):
        for i, j in mixer.edges:
            if state[i] == state[j]:
                continue
            swapped = state.copy()
            swapped[i], swapped[j] = swapped[j], swapped[i]
            b = index[tuple(int(v) for v in swapped)]
            h[a, b] = 1.0
    if not np.allclose(h, h.T, atol=1e-12):
        raise RuntimeError("constructed XY mixer is not Hermitian")
    return h
