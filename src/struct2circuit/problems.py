"""Cardinality-constrained QUBO problem definitions and deterministic generators."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int8]


def _validate_generator_parameters(n: int, k: int, density: float) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer of at least 2")
    if not isinstance(k, int) or isinstance(k, bool) or not 0 < k < n:
        raise ValueError("k must satisfy 0 < k < n")
    if not np.isfinite(density) or not 0.0 <= density <= 1.0:
        raise ValueError("density must lie in [0, 1]")


def _validate_weight_range(weight_range: tuple[float, float]) -> tuple[float, float]:
    if len(weight_range) != 2:
        raise ValueError("weight_range must contain two values")
    low, high = (float(value) for value in weight_range)
    if not np.isfinite([low, high]).all() or low <= 0.0 or high < low:
        raise ValueError("weight_range must be finite and satisfy 0 < low <= high")
    return low, high


def _permutation_metadata(permutation: NDArray[np.int64]) -> dict[str, list[int]]:
    inverse = np.argsort(permutation)
    return {
        "permutation": permutation.astype(int).tolist(),
        "inverse_permutation": inverse.astype(int).tolist(),
    }


def _sample_weight(
    rng: np.random.Generator,
    distribution: str,
    low: float,
    high: float,
) -> float:
    if distribution == "uniform":
        return float(rng.uniform(low, high))
    if distribution == "log_uniform":
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    raise ValueError("weight_distribution must be 'uniform' or 'log_uniform'")


@dataclass(frozen=True)
class CardinalityQUBO:
    """Minimize ``x.T @ Q @ x + c.T @ x`` subject to ``sum(x) == k``."""

    Q: FloatArray
    c: FloatArray
    k: int
    name: str = "cardinality_qubo"
    metadata: dict | None = None

    def __post_init__(self) -> None:
        q = np.asarray(self.Q, dtype=float)
        c = np.asarray(self.c, dtype=float)
        if q.ndim != 2 or q.shape[0] != q.shape[1]:
            raise ValueError("Q must be a square matrix")
        if c.shape != (q.shape[0],):
            raise ValueError("c must have one entry per binary variable")
        if not 0 < self.k < q.shape[0]:
            raise ValueError("k must satisfy 0 < k < n")
        if not np.allclose(q, q.T, atol=1e-12):
            raise ValueError("Q must be symmetric")
        object.__setattr__(self, "Q", q)
        object.__setattr__(self, "c", c)

    @property
    def n(self) -> int:
        return int(self.Q.shape[0])

    def feasible_basis(self) -> IntArray:
        """Enumerate binary strings of exact Hamming weight ``k`` lexicographically."""
        rows: list[np.ndarray] = []
        for occupied in combinations(range(self.n), self.k):
            row = np.zeros(self.n, dtype=np.int8)
            row[list(occupied)] = 1
            rows.append(row)
        return np.stack(rows)

    def costs(self, x: ArrayLike) -> FloatArray:
        """Evaluate one or many binary decisions."""
        arr = np.asarray(x, dtype=float)
        if arr.shape[-1] != self.n:
            raise ValueError("last dimension of x must equal n")
        quadratic = np.einsum("...i,ij,...j->...", arr, self.Q, arr)
        return np.asarray(quadratic + arr @ self.c, dtype=float)


def block_correlated_qubo(
    n: int,
    k: int,
    seed: int,
    *,
    n_blocks: int = 2,
    block_strength: float = 0.72,
    cross_strength: float = 0.16,
    linear_scale: float = 0.55,
) -> CardinalityQUBO:
    """Generate a PSD, block-correlated selection QUBO.

    The generator is a compact proxy for index-tracking/portfolio selection:
    ``Q`` represents pairwise co-movement and the linear term rewards individual
    tracking quality. Structural heterogeneity is controlled without using
    market data, making the pilot reproducible and license-free.
    """
    if n_blocks < 1 or n_blocks > n:
        raise ValueError("n_blocks must be between 1 and n")
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % n_blocks
    rng.shuffle(labels)

    loadings = np.zeros((n, n_blocks + 1), dtype=float)
    for i, block in enumerate(labels):
        loadings[i, block] = block_strength * rng.uniform(0.85, 1.15)
        loadings[i, -1] = cross_strength * rng.uniform(0.75, 1.25)
    covariance = loadings @ loadings.T
    covariance += np.diag(rng.uniform(0.18, 0.34, size=n))
    scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    correlation = covariance / scale

    quality = rng.normal(loc=0.65, scale=0.18, size=n)
    quality += rng.normal(scale=0.07, size=n_blocks)[labels]
    c = -linear_scale * quality

    return CardinalityQUBO(
        Q=correlation,
        c=c,
        k=k,
        name=f"block_corr_n{n}_k{k}_seed{seed}",
        metadata={
            "generator_version": "block_correlated_v1",
            "seed": int(seed),
            "n_blocks": int(n_blocks),
            "block_strength": float(block_strength),
            "cross_strength": float(cross_strength),
            "linear_scale": float(linear_scale),
            "labels": labels.tolist(),
        },
    )


def weighted_densest_k_subgraph_qubo(
    n: int,
    k: int,
    density: float,
    seed: int,
    *,
    weight_range: tuple[float, float] = (0.5, 1.5),
    weight_distribution: str = "uniform",
    planted_community_strength: float = 1.0,
) -> CardinalityQUBO:
    """Generate a weighted densest-k-subgraph minimization QUBO.

    Vertices are permuted after generation. ``permutation[new]`` gives the
    corresponding pre-permutation vertex; its inverse is also recorded.
    """
    _validate_generator_parameters(n, k, density)
    low, high = _validate_weight_range(weight_range)
    if weight_distribution not in {"uniform", "log_uniform"}:
        raise ValueError("weight_distribution must be 'uniform' or 'log_uniform'")
    if not np.isfinite(planted_community_strength) or planted_community_strength <= 0.0:
        raise ValueError("planted_community_strength must be positive and finite")
    rng = np.random.default_rng(seed)
    community = np.arange(n) < max(k, n // 3)
    weights = np.zeros((n, n), dtype=float)
    for i, j in combinations(range(n), 2):
        if rng.random() < density:
            factor = planted_community_strength if community[i] and community[j] else 1.0
            weights[i, j] = weights[j, i] = _sample_weight(rng, weight_distribution, low, high) * factor
    permutation = rng.permutation(n)
    weights = weights[np.ix_(permutation, permutation)]
    q = -weights / 2.0
    metadata = {
        "generator_version": "weighted_densest_k_subgraph_v1",
        "seed": int(seed),
        "density": float(density),
        "weight_range": [low, high],
        "weight_distribution": weight_distribution,
        "planted_community_strength": float(planted_community_strength),
        "planted_community_size": int(max(k, n // 3)),
        "weights": weights.tolist(),
        **_permutation_metadata(permutation),
    }
    return CardinalityQUBO(q, np.zeros(n), k, f"densest_k_n{n}_k{k}_seed{seed}", metadata)


def weighted_max_k_vertex_cover_qubo(
    n: int,
    k: int,
    density: float,
    seed: int,
    *,
    weight_range: tuple[float, float] = (0.5, 1.5),
    weight_distribution: str = "uniform",
    community_strength: float = 1.0,
    hub_strength: float = 1.0,
) -> CardinalityQUBO:
    """Generate a weighted maximum-k-vertex-cover QUBO with zero diagonal."""
    _validate_generator_parameters(n, k, density)
    low, high = _validate_weight_range(weight_range)
    if weight_distribution not in {"uniform", "log_uniform"}:
        raise ValueError("weight_distribution must be 'uniform' or 'log_uniform'")
    for value, name in ((community_strength, "community_strength"), (hub_strength, "hub_strength")):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    rng = np.random.default_rng(seed)
    community = np.arange(n) < max(2, n // 2)
    weights = np.zeros((n, n), dtype=float)
    for i, j in combinations(range(n), 2):
        if rng.random() < density:
            factor = community_strength if community[i] and community[j] else 1.0
            if i == 0 or j == 0:
                factor *= hub_strength
            weights[i, j] = weights[j, i] = _sample_weight(rng, weight_distribution, low, high) * factor
    permutation = rng.permutation(n)
    weights = weights[np.ix_(permutation, permutation)]
    q = weights / 2.0
    np.fill_diagonal(q, 0.0)
    c = -weights.sum(axis=1)
    metadata = {
        "generator_version": "weighted_max_k_vertex_cover_v1",
        "seed": int(seed),
        "density": float(density),
        "weight_range": [low, high],
        "weight_distribution": weight_distribution,
        "community_strength": float(community_strength),
        "hub_strength": float(hub_strength),
        "diagonal_convention": "zero",
        "weights": weights.tolist(),
        **_permutation_metadata(permutation),
    }
    return CardinalityQUBO(q, c, k, f"max_k_cover_n{n}_k{k}_seed{seed}", metadata)


def weak_structure_null_qubo(
    n: int,
    k: int,
    density: float,
    coefficient_scale: float,
    seed: int,
) -> CardinalityQUBO:
    """Generate an exchangeable weak-structure negative-control QUBO.

    Present off-diagonal entries and linear terms are independent centered
    Gaussians. Both are divided by ``sqrt(max(1, density * (n - 1)))`` so the
    typical aggregate interaction scale remains comparable as ``n`` changes.
    This is not a proof that an instance contains no exploitable information.
    """
    _validate_generator_parameters(n, k, density)
    if not np.isfinite(coefficient_scale) or coefficient_scale <= 0.0:
        raise ValueError("coefficient_scale must be positive and finite")
    rng = np.random.default_rng(seed)
    normalization = float(np.sqrt(max(1.0, density * (n - 1))))
    q = np.zeros((n, n), dtype=float)
    for i, j in combinations(range(n), 2):
        if rng.random() < density:
            q[i, j] = q[j, i] = rng.normal() * coefficient_scale / normalization
    c = rng.normal(size=n) * coefficient_scale / normalization
    permutation = rng.permutation(n)
    q = q[np.ix_(permutation, permutation)]
    c = c[permutation]
    metadata = {
        "generator_version": "weak_structure_null_v1",
        "seed": int(seed),
        "density": float(density),
        "coefficient_scale": float(coefficient_scale),
        "normalization": normalization,
        "coefficient_distribution": "zero_mean_gaussian",
        "scale_normalization": "sqrt(max(1,density*(n-1)))",
        **_permutation_metadata(permutation),
    }
    return CardinalityQUBO(q, c, k, f"weak_null_n{n}_k{k}_seed{seed}", metadata)
