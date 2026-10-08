"""Regression checks for spectral references, without performance claims."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from struct2circuit.mixers import MixerSpec, complete_mixer, ring_mixer  # noqa: E402
from struct2circuit.problems import CardinalityQUBO  # noqa: E402
from struct2circuit.simulator import FeasibleSubspaceQAOA  # noqa: E402


class SpectralReferenceTests(unittest.TestCase):
    @staticmethod
    def problem(n: int, k: int) -> CardinalityQUBO:
        return CardinalityQUBO(np.zeros((n, n)), np.arange(n, dtype=float), k)

    @staticmethod
    def star() -> MixerSpec:
        return MixerSpec("star", 5, ((0, 1), (0, 2), (0, 3), (0, 4)), "test fixture")

    @staticmethod
    def irregular() -> MixerSpec:
        return MixerSpec(
            "irregular",
            6,
            ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2), (1, 4)),
            "test fixture",
        )

    def cases(self) -> tuple[tuple[MixerSpec, int], ...]:
        return (
            (self.star(), 1),
            (ring_mixer(5), 1),
            (complete_mixer(5), 2),
            (complete_mixer(6), 3),
            (ring_mixer(6), 3),
            (self.irregular(), 3),
        )

    def permuted_simulator(
        self, simulator: FeasibleSubspaceQAOA, permutation: np.ndarray
    ) -> tuple[FeasibleSubspaceQAOA, np.ndarray]:
        """New coordinate r denotes old coordinate permutation[r]."""
        inverse = np.argsort(permutation)
        problem = simulator.problem
        permuted_problem = CardinalityQUBO(
            problem.Q[np.ix_(permutation, permutation)],
            problem.c[permutation],
            problem.k,
        )
        permuted_mixer = MixerSpec(
            "permuted",
            problem.n,
            tuple((int(inverse[i]), int(inverse[j])) for i, j in simulator.mixer.edges),
            "relabeling of test fixture",
        )
        permuted = FeasibleSubspaceQAOA(permuted_problem, permuted_mixer)
        original_indices = {tuple(row): index for index, row in enumerate(simulator.basis)}
        original_order = np.array(
            [original_indices[tuple(row[inverse])] for row in permuted.basis]
        )
        return permuted, original_order

    def test_both_extrema_are_normalized_eigenstates(self) -> None:
        for mixer, k in self.cases():
            for which in ("low", "high"):
                with self.subTest(mixer=mixer.name, n=mixer.n, k=k, extremum=which):
                    sim = FeasibleSubspaceQAOA(
                        self.problem(mixer.n, k), mixer, initialization=f"mixer_{which}"
                    )
                    eigenvalues = np.linalg.eigvalsh(sim.mixer_hamiltonian)
                    target = eigenvalues[0 if which == "low" else -1]
                    self.assertAlmostEqual(np.linalg.norm(sim.initial_state), 1.0, places=12)
                    self.assertLess(
                        np.linalg.norm(sim.mixer_hamiltonian @ sim.initial_state - target * sim.initial_state),
                        1e-12,
                    )

    def test_uniform_extremal_fidelities_match_analytic_values(self) -> None:
        # The star's two extremal vectors give overlaps 1/10 and 9/10.
        # Complete token graphs are regular, so uniform is exactly their PF state.
        # A physical ring need not have a regular configuration graph at interior k.
        cases = (
            (self.star(), 1, 0.1, 0.9),
            (complete_mixer(5), 2, 0.0, 1.0),
            (complete_mixer(6), 3, 0.0, 1.0),
            (ring_mixer(5), 1, 0.0, 1.0),
            (ring_mixer(6), 3, 0.0, 0.9),
        )
        for mixer, k, low, high in cases:
            sim = FeasibleSubspaceQAOA(self.problem(mixer.n, k), mixer)
            for which, expected in (("low", low), ("high", high)):
                with self.subTest(mixer=mixer.name, n=mixer.n, k=k, extremum=which):
                    self.assertAlmostEqual(
                        sim.uniform_mixer_extremal_fidelity(which), expected, places=12
                    )
                    state = getattr(sim, f"mixer_{which}_state")
                    self.assertAlmostEqual(
                        abs(np.vdot(sim.uniform_state, state)) ** 2, expected, places=12
                    )

    def test_degenerate_references_are_repeatable(self) -> None:
        for mixer, k, multiplicity in (
            (ring_mixer(5), 1, 2),
            (complete_mixer(5), 2, 5),
            (complete_mixer(6), 3, 5),
        ):
            with self.subTest(mixer=mixer.name, n=mixer.n, k=k):
                problem = self.problem(mixer.n, k)
                first = FeasibleSubspaceQAOA(problem, mixer)
                second = FeasibleSubspaceQAOA(problem, mixer)
                self.assertEqual(int(first._extremal_mask("low").sum()), multiplicity)
                self.assertLess(first.uniform_mixer_extremal_fidelity("low"), 1e-24)
                for which in ("low", "high"):
                    np.testing.assert_allclose(
                        getattr(first, f"mixer_{which}_state"),
                        getattr(second, f"mixer_{which}_state"),
                        atol=1e-14,
                        rtol=0.0,
                    )

    def test_reference_and_fidelity_are_invariant_to_eigenspace_basis_rotation(self) -> None:
        rng = np.random.default_rng(901)
        for mixer, k in self.cases():
            sim = FeasibleSubspaceQAOA(self.problem(mixer.n, k), mixer)
            original_vectors = sim._mixer_eigenvectors.astype(complex)
            for which in ("low", "high"):
                mask = sim._extremal_mask(which)
                eigenspace = original_vectors[:, mask]
                reference = getattr(sim, f"mixer_{which}_state")
                fidelity = sim.uniform_mixer_extremal_fidelity(which)
                for repetition in range(5):
                    with self.subTest(mixer=mixer.name, n=mixer.n, k=k, extremum=which, rotation=repetition):
                        dimension = eigenspace.shape[1]
                        random_matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
                            size=(dimension, dimension)
                        )
                        rotation, _ = np.linalg.qr(random_matrix)
                        sim._mixer_eigenvectors = original_vectors.copy()
                        sim._mixer_eigenvectors[:, mask] = eigenspace @ rotation
                        rotated_reference = sim._spectral_reference(which)
                        # A tied phase pivot can change global phase under roundoff;
                        # physical-state fidelity, not raw vector equality, is invariant.
                        self.assertAlmostEqual(
                            abs(np.vdot(reference, rotated_reference)) ** 2, 1.0, places=12
                        )
                        self.assertAlmostEqual(
                            sim.uniform_mixer_extremal_fidelity(which), fidelity, places=12
                        )
                sim._mixer_eigenvectors = original_vectors.copy()

    def test_nonzero_uniform_projection_fixes_its_phase(self) -> None:
        sim = FeasibleSubspaceQAOA(self.problem(5, 1), self.star())
        for which in ("low", "high"):
            with self.subTest(extremum=which):
                overlap = np.vdot(sim.uniform_state, getattr(sim, f"mixer_{which}_state"))
                self.assertGreater(overlap.real, 0.0)
                self.assertAlmostEqual(overlap.imag, 0.0, places=14)

    def test_fallback_fixes_phase_for_an_unambiguous_pivot(self) -> None:
        sim = FeasibleSubspaceQAOA(self.problem(5, 2), complete_mixer(5))
        state = sim.mixer_low_state
        self.assertLess(sim.uniform_mixer_extremal_fidelity("low"), 1e-24)
        pivot = int(np.argmax(np.abs(state)))
        magnitudes = np.sort(np.abs(state))
        self.assertGreater(magnitudes[-1] - magnitudes[-2], 0.1)
        self.assertGreater(state[pivot].real, 0.0)
        self.assertAlmostEqual(state[pivot].imag, 0.0, places=14)

    def test_high_is_the_unique_positive_perron_reference_and_is_equivariant(self) -> None:
        for mixer, k in self.cases():
            with self.subTest(mixer=mixer.name, n=mixer.n, k=k):
                sim = FeasibleSubspaceQAOA(self.problem(mixer.n, k), mixer)
                self.assertEqual(int(sim._extremal_mask("high").sum()), 1)
                self.assertTrue(np.all(sim.mixer_high_state.real > 0.0))
                np.testing.assert_allclose(sim.mixer_high_state.imag, 0.0, atol=1e-14)
                permutation = np.roll(np.arange(mixer.n), 1)
                permuted, original_order = self.permuted_simulator(sim, permutation)
                np.testing.assert_allclose(
                    permuted.mixer_high_state,
                    sim.mixer_high_state[original_order],
                    atol=1e-12,
                    rtol=0.0,
                )

    def test_degenerate_low_fallback_can_depend_on_labels(self) -> None:
        # A cyclic relabeling preserves these graphs but changes the selected seed.
        # Deterministic reference selection must not be mistaken for equivariance.
        for mixer, k, expected_fidelity in (
            (ring_mixer(5), 1, (3.0 + np.sqrt(5.0)) / 8.0),
            (complete_mixer(5), 2, 1.0 / 9.0),
        ):
            with self.subTest(mixer=mixer.name, k=k):
                sim = FeasibleSubspaceQAOA(self.problem(mixer.n, k), mixer)
                permutation = np.roll(np.arange(mixer.n), 1)
                permuted, original_order = self.permuted_simulator(sim, permutation)
                fidelity = abs(
                    np.vdot(permuted.mixer_low_state, sim.mixer_low_state[original_order])
                ) ** 2
                self.assertAlmostEqual(fidelity, expected_fidelity, places=12)
                self.assertLess(fidelity, 0.9)


if __name__ == "__main__":
    unittest.main()
