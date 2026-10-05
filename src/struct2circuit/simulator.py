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
    """Exact feasible-subspace simulator with explicit initialization control.

    ``initialization="uniform"`` is the common baseline used by the pilot.
    ``initialization="mixer_low"`` and ``"mixer_high"`` select deterministic
    states from the lowest/highest mixer eigenspaces. Both spectral extrema are
    exposed because which one represents an aligned reference depends on the
    mixer sign convention. These are diagnostics, not hardware state-preparation
    claims.
    """

    def __init__(
        self,
        problem: CardinalityQUBO,
        mixer: MixerSpec,
        *,
        initialization: str = "uniform",
    ) -> None:
        if problem.n != mixer.n:
            raise ValueError("problem and mixer sizes differ")
        if initialization not in {"uniform", "mixer_low", "mixer_high"}:
            raise ValueError(
                "initialization must be 'uniform', 'mixer_low', or 'mixer_high'"
            )
        self.problem = problem
        self.mixer = mixer
        self.initialization = initialization
        self.basis = problem.feasible_basis()
        self.costs = problem.costs(self.basis)
        self.cost_min = float(np.min(self.costs))
        self.cost_max = float(np.max(self.costs))
        self.optimal_mask = np.isclose(self.costs, self.cost_min, rtol=0.0, atol=1e-10)
        self.mixer_hamiltonian = xy_mixer_hamiltonian(self.basis, mixer)
        self._mixer_eigenvalues, self._mixer_eigenvectors = np.linalg.eigh(
            self.mixer_hamiltonian
        )

        self.uniform_state = np.full(
            len(self.basis), 1.0 / np.sqrt(len(self.basis)), dtype=complex
        )
        self.mixer_low_state = self._spectral_reference("low")
        self.mixer_high_state = self._spectral_reference("high")

        states = {
            "uniform": self.uniform_state,
            "mixer_low": self.mixer_low_state,
            "mixer_high": self.mixer_high_state,
        }
        self.initial_state = states[initialization]

    def _extremal_mask(self, which: str) -> NDArray[np.bool_]:
        target = (
            self._mixer_eigenvalues[0]
            if which == "low"
            else self._mixer_eigenvalues[-1]
        )
        return np.isclose(self._mixer_eigenvalues, target, rtol=0.0, atol=1e-10)

    def _spectral_reference(self, which: str) -> ComplexArray:
        """Choose a deterministic state from an extremal mixer eigenspace.

        If the uniform state has nonzero projection into the eigenspace, use the
        normalized projection. Otherwise resolve degeneracy deterministically by
        projecting computational-basis vectors in lexicographic order.
        """
        eigenspace = self._mixer_eigenvectors[:, self._extremal_mask(which)]
        projected = eigenspace @ (eigenspace.conj().T @ self.uniform_state)
        norm = float(np.linalg.norm(projected))
        if norm > 1e-12:
            return np.asarray(projected / norm, dtype=complex)

        for index in range(self.feasible_dimension):
            seed = np.zeros(self.feasible_dimension, dtype=complex)
            seed[index] = 1.0
            projected = eigenspace @ (eigenspace.conj().T @ seed)
            norm = float(np.linalg.norm(projected))
            if norm > 1e-12:
                state = projected / norm
                pivot = int(np.argmax(np.abs(state)))
                phase = np.angle(state[pivot])
                return np.asarray(state * np.exp(-1j * phase), dtype=complex)
        raise RuntimeError("could not construct deterministic extremal mixer state")

    def uniform_mixer_extremal_fidelity(self, which: str) -> float:
        """Return uniform-state fidelity with a low/high mixer eigenspace."""
        if which not in {"low", "high"}:
            raise ValueError("which must be 'low' or 'high'")
        eigenspace = self._mixer_eigenvectors[:, self._extremal_mask(which)]
        overlap = eigenspace.conj().T @ self.uniform_state
        return float(np.sum(np.abs(overlap) ** 2))

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
