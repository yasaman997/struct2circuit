"""Cardinality-constrained QUBO problem definitions and deterministic generators."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int8]


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
            "seed": int(seed),
            "n_blocks": int(n_blocks),
            "block_strength": float(block_strength),
            "cross_strength": float(cross_strength),
            "labels": labels.tolist(),
        },
    )

