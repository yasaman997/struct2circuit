from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from struct2circuit.mixers import (  # noqa: E402
    complete_mixer,
    graph_connected,
    ring_mixer,
    structure_conditioned_mixer,
    xy_mixer_hamiltonian,
)
from struct2circuit.optimize import optimize_p1  # noqa: E402
from struct2circuit.problems import block_correlated_qubo  # noqa: E402
from struct2circuit.simulator import FeasibleSubspaceQAOA  # noqa: E402


class InvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = block_correlated_qubo(7, 3, 42)

    def test_basis_has_exact_cardinality(self) -> None:
        basis = self.problem.feasible_basis()
        self.assertTrue(np.all(basis.sum(axis=1) == self.problem.k))
        self.assertEqual(len(basis), 35)

    def test_all_mixer_graphs_are_connected(self) -> None:
        mixers = [
            ring_mixer(self.problem.n),
            structure_conditioned_mixer(self.problem.Q, self.problem.n),
            complete_mixer(self.problem.n),
        ]
        self.assertTrue(all(graph_connected(m.n, list(m.edges)) for m in mixers))

    def test_xy_hamiltonian_is_hermitian_and_stays_in_basis(self) -> None:
        basis = self.problem.feasible_basis()
        h = xy_mixer_hamiltonian(basis, structure_conditioned_mixer(self.problem.Q, self.problem.n))
        self.assertTrue(np.allclose(h, h.T.conj()))
        self.assertEqual(h.shape, (len(basis), len(basis)))

    def test_qaoa_state_is_normalized_and_feasible(self) -> None:
        sim = FeasibleSubspaceQAOA(self.problem, ring_mixer(self.problem.n))
        result = sim.evaluate([0.73, 1.21], [0.41, 0.89])
        self.assertAlmostEqual(result.state_norm, 1.0, places=11)
        self.assertAlmostEqual(result.feasibility_probability, 1.0, places=11)

    def test_structure_mixer_is_deterministic(self) -> None:
        a = structure_conditioned_mixer(self.problem.Q, self.problem.n)
        b = structure_conditioned_mixer(self.problem.Q, self.problem.n)
        self.assertEqual(a.edges, b.edges)

    def test_optimizer_is_deterministic(self) -> None:
        sim = FeasibleSubspaceQAOA(self.problem, ring_mixer(self.problem.n))
        a = optimize_p1(sim, grid_size=7)
        b = optimize_p1(sim, grid_size=7)
        self.assertAlmostEqual(a.result.expectation, b.result.expectation, places=10)

    def test_mixer_ground_initialization_is_normalized_and_feasible(self) -> None:
        sim = FeasibleSubspaceQAOA(
            self.problem,
            structure_conditioned_mixer(self.problem.Q, self.problem.n),
            initialization="mixer_ground",
        )
        self.assertAlmostEqual(np.linalg.norm(sim.initial_state), 1.0, places=11)
        self.assertAlmostEqual(np.sum(np.abs(sim.initial_state) ** 2), 1.0, places=11)
        self.assertGreaterEqual(sim.uniform_mixer_ground_state_fidelity, 0.0)
        self.assertLessEqual(sim.uniform_mixer_ground_state_fidelity, 1.0)

    def test_uniform_and_aligned_initializations_are_explicitly_distinct_controls(self) -> None:
        mixer = structure_conditioned_mixer(self.problem.Q, self.problem.n)
        uniform = FeasibleSubspaceQAOA(self.problem, mixer, initialization="uniform")
        aligned = FeasibleSubspaceQAOA(self.problem, mixer, initialization="mixer_ground")
        self.assertTrue(np.allclose(uniform.initial_state, uniform.uniform_state))
        self.assertTrue(np.allclose(aligned.initial_state, aligned.mixer_ground_state))


if __name__ == "__main__":
    unittest.main()
