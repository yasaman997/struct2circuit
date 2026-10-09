"""Mixer graphs and exact Hamming-weight-preserving XY Hamiltonians."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import fsum, gcd, hypot, sqrt

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


@dataclass(frozen=True)
class ExchangeProfileStatistics:
    """Conditional exchange moments, with zero diagonals.

    ``mean[i, j]`` is directed from occupied i to unoccupied j and is
    antisymmetric. ``variance`` and ``rms`` are symmetric edge descriptors.
    """

    mean: FloatArray
    variance: FloatArray
    rms: FloatArray


def exchange_profile_statistics(
    Q: FloatArray, c: FloatArray, k: int
) -> ExchangeProfileStatistics:
    """Exact exchange moments under the uniform fixed-weight feasible prior.

    Condition on x[i] = 1, x[j] = 0, and sum(x) = k. With N = n - 2,
    m = k - 1, a[l] = Q[j,l] - Q[i,l] for l outside {i,j}, and
    d0 = Q[j,j] - Q[i,i] + c[j] - c[i], the exchange cost is
    d0 + 2*sum(a[l]*x[l]). Its mean is d0 + 2*m*mean(a), and its
    variance is 4*m*(N-m)/(N-1)*mean((a-mean(a))**2) for N > 1.
    The N <= 1 and endpoint-cardinality cases have zero variance.

    These identities assume exactly symmetric Q. Validation shares
    CardinalityQUBO's absolute tolerance of 1e-12 with no relative tolerance;
    inputs are not symmetrized. For accepted asymmetry eps=max(abs(Q-Q.T)),
    the formula's swap costs, conditional mean, and RMS can differ from those
    of x.T@Q@x by up to 2*(k-1)*eps, apart from floating-point evaluation.
    This is an absolute bound, not a relative guarantee for tiny objectives.

    For the mean, reduce 2*m/N to integers p/d and evaluate
    (d*d0 + p*sum(a[l]))/d. All original coefficients enter one compensated
    sum, with integer weights expressed as repeated terms; division follows
    cancellation. No rounded row difference or centered aggregate enters the
    mean. At m=N, p=2 and d=1 give the full deterministic exchange directly.
    Variance uses a reference remaining site r and computes a[l] - a[r] as
    one compensated sum of four original coefficients. Compensated comparisons
    select the reference. Input rounding, summation range, and final floating-
    point rounding remain numerical limits.

    These moments use only Q, c, and k, without solving the optimization
    problem. The RMS describes a uniform feasible prior, not the transition
    distribution of an optimized or spectrally initialized quantum state.
    """
    q = np.asarray(Q, dtype=float)
    linear = np.asarray(c, dtype=float)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("Q must be square")
    n = q.shape[0]
    if n < 2:
        raise ValueError("Q must describe at least 2 variables")
    if linear.shape != (n,):
        raise ValueError("c must have one entry per variable")
    if not np.isfinite(q).all() or not np.isfinite(linear).all():
        raise ValueError("Q and c must be finite")
    if not np.allclose(q, q.T, rtol=0.0, atol=1e-12):
        raise ValueError("Q must be symmetric")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or not 0 < k < n:
        raise ValueError("k must be an integer satisfying 0 < k < n")

    remaining_count = n - 2
    occupied_count = int(k) - 1
    mean_denominator, interaction_copies = 1, 0
    if occupied_count:
        divisor = gcd(2 * occupied_count, remaining_count)
        mean_denominator = remaining_count // divisor
        interaction_copies = 2 * occupied_count // divisor
    means = np.zeros((n, n), dtype=float)
    variances = np.zeros((n, n), dtype=float)
    rms = np.zeros((n, n), dtype=float)
    for i, j in combinations(range(n), 2):
        others = [ell for ell in range(n) if ell not in (i, j)]
        # Keep every original coefficient until all mean contributions cancel.
        # Repeated terms avoid rounding integer products before that sum.
        mean_terms = [q[j, j], -q[i, i], linear[j], -linear[i]] * mean_denominator
        if interaction_copies:
            for ell in others:
                mean_terms.extend([q[j, ell], -q[i, ell]] * interaction_copies)
        mean = fsum(mean_terms) / mean_denominator
        standard_deviation = 0.0
        if 0 < occupied_count < remaining_count:
            # Choose the minimum a[l] without first rounding a large common
            # offset into each a[l]. True reference ties give identical centered
            # differences, so the reference choice does not add a label rule.
            reference = others[0]
            for ell in others[1:]:
                if fsum((q[j, ell], -q[i, ell],
                         -q[j, reference], q[i, reference])) < 0.0:
                    reference = ell
            centered = np.asarray([
                fsum((q[j, ell], -q[i, ell], -q[j, reference], q[i, reference]))
                for ell in others
            ])
            centered_mean = fsum(centered) / remaining_count
            standard_deviation = hypot(*(centered - centered_mean)) * sqrt(
                4.0 * occupied_count * (remaining_count - occupied_count)
                / ((remaining_count - 1) * remaining_count)
            )
        variance = standard_deviation * standard_deviation
        means[i, j], means[j, i] = mean, -mean
        variances[i, j] = variances[j, i] = variance
        rms[i, j] = rms[j, i] = hypot(mean, standard_deviation)
    return ExchangeProfileStatistics(means, variances, rms)


def exchange_profile_scores(Q: FloatArray, c: FloatArray, k: int) -> FloatArray:
    """Return exact conditional RMS exchange costs as symmetric edge scores.

    Low/high RMS are competing mechanism hypotheses, not established claims
    of better QAOA performance. Mean and variance remain separately available
    through :func:`exchange_profile_statistics`.
    """
    return exchange_profile_statistics(Q, c, k).rms


def score_conditioned_mixer(
    scores: FloatArray,
    edge_budget: int,
    *,
    prefer: str = "low",
    name: str = "score_conditioned",
) -> MixerSpec:
    """Build a connected sparse mixer from a predeclared symmetric edge score.

    Build a minimum/maximum spanning tree for low/high scores, then add ranked
    remaining edges. Exact score ties are broken lexicographically by variable
    labels. This is deterministic and permutation-equivariant for distinct
    scores, but exact ties can make the result depend on the labels.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError("scores must be square")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if not np.allclose(s, s.T, rtol=0.0, atol=1e-12):
        raise ValueError("scores must be symmetric")
    if prefer not in {"low", "high"}:
        raise ValueError("prefer must be 'low' or 'high'")
    n = s.shape[0]
    if n < 2:
        raise ValueError("scores must describe at least 2 variables")
    if not isinstance(edge_budget, (int, np.integer)) or isinstance(edge_budget, bool):
        raise ValueError("edge_budget must be an integer")
    maximum = n * (n - 1) // 2
    if not n - 1 <= edge_budget <= maximum:
        raise ValueError(f"edge_budget must lie in [{n - 1}, {maximum}]")
    sign = 1.0 if prefer == "low" else -1.0
    ranked = sorted(
        ((sign * float(s[i, j]), i, j) for i, j in combinations(range(n), 2)),
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
    return MixerSpec(name, n, tuple(chosen), f"{prefer}-score spanning tree plus ranked remaining edges")


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
