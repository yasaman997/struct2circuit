"""Exact QAOA simulation in the cardinality-feasible subspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .mixers import MixerSpec, xy_mixer_hamiltonian
from .problems import CardinalityQUBO


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class QAOAResult:
    expectation: float
    normalized_gap: float
    probability_optimum: float
    state_norm: float
    feasibility_probability: float


class FeasibleSubspaceQAOA:
    """A small-system simulator with exact feasibility by representation."""

    def __init__(self, problem: CardinalityQUBO, mixer: MixerSpec) -> None:
        if problem.n != mixer.n:
            raise ValueError("problem and mixer sizes differ")
        self.problem = problem
        self.mixer = mixer
        self.basis = problem.feasible_basis()
        self.costs = problem.costs(self.basis)
        self.cost_min = float(np.min(self.costs))
        self.cost_max = float(np.max(self.costs))
        self.optimal_mask = np.isclose(self.costs, self.cost_min, rtol=0.0, atol=1e-10)
        self.mixer_hamiltonian = xy_mixer_hamiltonian(self.basis, mixer)
        self._mixer_eigenvalues, self._mixer_eigenvectors = np.linalg.eigh(self.mixer_hamiltonian)
        self.initial_state = np.full(len(self.basis), 1.0 / np.sqrt(len(self.basis)), dtype=complex)

    @property
    def feasible_dimension(self) -> int:
        return len(self.basis)

    def _apply_mixer(self, state: ComplexArray, beta: float) -> ComplexArray:
        vecs = self._mixer_eigenvectors
        spectral = vecs.T.conj() @ state
        spectral *= np.exp(-1j * beta * self._mixer_eigenvalues)
        return np.asarray(vecs @ spectral, dtype=complex)

    def state(self, gamma: ArrayLike, beta: ArrayLike) -> ComplexArray:
        gammas = np.atleast_1d(np.asarray(gamma, dtype=float))
        betas = np.atleast_1d(np.asarray(beta, dtype=float))
        if gammas.shape != betas.shape or gammas.ndim != 1:
            raise ValueError("gamma and beta must be one-dimensional arrays of equal length")
        psi = self.initial_state.copy()
        for g, b in zip(gammas, betas, strict=True):
            psi *= np.exp(-1j * g * self.costs)
            psi = self._apply_mixer(psi, float(b))
        return psi

    def evaluate(self, gamma: ArrayLike, beta: ArrayLike) -> QAOAResult:
        psi = self.state(gamma, beta)
        probabilities = np.abs(psi) ** 2
        expectation = float(probabilities @ self.costs)
        span = self.cost_max - self.cost_min
        gap = 0.0 if span <= 1e-14 else (expectation - self.cost_min) / span
        return QAOAResult(
            expectation=expectation,
            normalized_gap=float(gap),
            probability_optimum=float(np.sum(probabilities[self.optimal_mask])),
            state_norm=float(np.linalg.norm(psi)),
            feasibility_probability=float(np.sum(probabilities)),
        )

