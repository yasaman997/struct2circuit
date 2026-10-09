"""Exact QAOA simulation in the cardinality-feasible subspace."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .mixers import MixerSpec, xy_mixer_hamiltonian
from .problems import CardinalityQUBO


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class QAOAResult:
    """QAOA outcomes for the declared numerical objective.

    ``expectation`` is the probability-weighted legacy NumPy cost evaluation,
    with separately evaluated quadratic and linear totals, in raw cost units.
    This remains true after dimensionless optimization. ``normalized_gap``
    averages centered, normalized compensated float64 costs instead; computing
    it from ``expectation`` and the compensated extrema can give a different
    answer through cancellation or rounding. It is not a raw-expectation gap.

    ``probability_optimum`` sums probability on exact minimum-equality states
    of the compensated float64 feasible costs used for normalization. It has
    no near-optimum tolerance; final cost-rounding ties remain numerical ties.
    Constant or unresolved costs withhold both normalized metrics as ``None``.
    """

    expectation: float
    normalized_gap: float | None
    probability_optimum: float | None
    state_norm: float
    feasibility_probability: float
    cost_status: str


class FeasibleSubspaceQAOA:
    """Exact feasible-subspace simulator with explicit initialization control.

    ``initialization="uniform"`` is the common baseline used by the pilot.
    ``initialization="mixer_low"`` and ``"mixer_high"`` select deterministic
    states from the lowest/highest mixer eigenspaces. For the implemented
    positive XY adjacency, the connected feasible graph has a unique positive
    Perron--Frobenius highest eigenstate (the ground state of ``-H_M``).
    The low state is an opposite-extremum sensitivity condition. Neither mode
    guarantees better optimization or constitutes a hardware preparation claim.
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
        if not np.all(np.isfinite(self.costs)):
            raise ValueError("feasible costs exceed the supported finite floating-point range")
        # Preserve raw costs for legacy phases and expectations. For normalized
        # dynamics, sum the represented coefficients together with compensated
        # partial sums: separately rounded quadratic/linear totals can otherwise
        # turn exact cancellation into a fictitious nonconstant landscape.
        # Binary occupation selects coefficients exactly, without products or
        # pre-addition of symmetric pairs that could introduce rounding first.
        accurate_costs = []
        try:
            for state in self.basis:
                occupied = np.flatnonzero(state)
                accurate_costs.append(fsum(
                    [float(problem.Q[i, j]) for i in occupied for j in occupied]
                    + [float(problem.c[i]) for i in occupied]
                ))
        except OverflowError as error:
            raise ValueError(
                "accurate feasible costs exceed the supported floating-point range"
            ) from error
        normalization_costs = np.asarray(accurate_costs)
        self.cost_min = float(np.min(normalization_costs))
        self.cost_max = float(np.max(normalization_costs))
        self.feasible_span = self.cost_max - self.cost_min
        if not np.isfinite(self.feasible_span):
            raise ValueError("feasible cost span exceeds the supported floating-point range")
        # No absolute raw-cost cutoff. If even accurate summation cannot resolve
        # distinct float64 costs, withhold normalized metrics: true constancy and
        # variation lost in input representation/final rounding remain combined.
        self.cost_status = (
            "nonconstant" if self.feasible_span > 0.0 else "constant_or_unresolved"
        )
        self.normalized_costs = (
            (normalization_costs - self.cost_min) / self.feasible_span
            if self.feasible_span > 0.0 else None
        )
        self.optimal_mask = (
            # Compare before normalization: division can round a positive gap
            # to zero. Equality here retains only minima of the same accurately
            # summed float64 cost representation used for normalization.
            normalization_costs == self.cost_min
            if self.normalized_costs is not None else np.zeros(len(self.costs), dtype=bool)
        )
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
        projecting computational-basis vectors in lexicographic order. The
        fallback is independent of the eigenvector basis up to phase, but can
        depend on vertex labels; determinism does not imply permutation equivariance.
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
        return self._evaluate_state(psi)

    def state_dimensionless(self, u: ArrayLike, beta: ArrayLike) -> ComplexArray:
        """Apply cost phases using ``u = gamma * feasible_span``.

        Centering costs removes an irrelevant global phase and normalizing them
        avoids raw-unit dependence in the new optimizer's coordinates. A zero
        accurately summed span has no resolved cost dynamics and uses the identity phase.
        ``state`` retains its original raw-phase arithmetic for legacy searches.
        """
        us = np.atleast_1d(np.asarray(u, dtype=float))
        betas = np.atleast_1d(np.asarray(beta, dtype=float))
        if us.shape != betas.shape or us.ndim != 1:
            raise ValueError("u and beta must be one-dimensional arrays of equal length")
        psi = self.initial_state.copy()
        for coordinate, b in zip(us, betas, strict=True):
            if self.normalized_costs is not None:
                psi *= np.exp(-1j * coordinate * self.normalized_costs)
            psi = self._apply_mixer(psi, float(b))
        return psi

    def evaluate_dimensionless(self, u: ArrayLike, beta: ArrayLike) -> QAOAResult:
        """Evaluate normalized phases without round-tripping through physical gamma."""
        return self._evaluate_state(self.state_dimensionless(u, beta))

    def _evaluate_state(self, psi: ComplexArray) -> QAOAResult:
        probabilities = np.abs(psi) ** 2
        # Keep the historical raw expectation arithmetic used by legacy search.
        expectation = float(probabilities @ self.costs)
        gap = (
            float(probabilities @ self.normalized_costs)
            if self.normalized_costs is not None else None
        )
        probability_optimum = (
            float(np.sum(probabilities[self.optimal_mask]))
            if self.normalized_costs is not None else None
        )
        return QAOAResult(
            expectation=expectation,
            normalized_gap=gap,
            probability_optimum=probability_optimum,
            state_norm=float(np.linalg.norm(psi)),
            feasibility_probability=float(np.sum(probabilities)),
            cost_status=self.cost_status,
        )
