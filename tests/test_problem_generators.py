from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from struct2circuit.problems import (  # noqa: E402
    weak_structure_null_qubo,
    weighted_densest_k_subgraph_qubo,
    weighted_max_k_vertex_cover_qubo,
)


class ProblemGeneratorTests(unittest.TestCase):
    def _assert_valid_replay(self, generator, *args, **kwargs) -> None:
        first = generator(*args, **kwargs)
        second = generator(*args, **kwargs)
        self.assertTrue(np.array_equal(first.Q, second.Q))
        self.assertTrue(np.array_equal(first.c, second.c))
        self.assertTrue(np.isfinite(first.Q).all() and np.isfinite(first.c).all())
        self.assertTrue(np.array_equal(first.Q, first.Q.T))
        permutation = np.asarray(first.metadata["permutation"])
        inverse = np.asarray(first.metadata["inverse_permutation"])
        self.assertTrue(np.array_equal(permutation[inverse], np.arange(first.n)))

    def test_densest_objective_equivalence_exhaustive(self) -> None:
        problem = weighted_densest_k_subgraph_qubo(6, 3, 0.7, 17, planted_community_strength=1.8)
        weights = np.asarray(problem.metadata["weights"])
        for x, energy in zip(problem.feasible_basis(), problem.costs(problem.feasible_basis()), strict=True):
            original = sum(weights[i, j] * x[i] * x[j] for i in range(6) for j in range(i + 1, 6))
            self.assertAlmostEqual(float(energy), -float(original), places=12)

    def test_cover_objective_equivalence_exhaustive(self) -> None:
        problem = weighted_max_k_vertex_cover_qubo(6, 2, 0.65, 29, community_strength=1.4, hub_strength=1.6)
        weights = np.asarray(problem.metadata["weights"])
        self.assertTrue(np.array_equal(np.diag(problem.Q), np.zeros(6)))
        for x, energy in zip(problem.feasible_basis(), problem.costs(problem.feasible_basis()), strict=True):
            original = sum(
                weights[i, j] * (x[i] + x[j] - x[i] * x[j])
                for i in range(6) for j in range(i + 1, 6)
            )
            self.assertAlmostEqual(float(energy), -float(original), places=12)

    def test_replay_validity_and_permutations(self) -> None:
        self._assert_valid_replay(weighted_densest_k_subgraph_qubo, 7, 3, 0.5, 4)
        self._assert_valid_replay(weighted_max_k_vertex_cover_qubo, 7, 3, 0.5, 4)
        self._assert_valid_replay(weak_structure_null_qubo, 7, 3, 0.5, 1.2, 4)

    def test_null_documented_construction(self) -> None:
        problem = weak_structure_null_qubo(8, 3, 0.5, 2.0, 8)
        self.assertEqual(problem.metadata["generator_version"], "weak_structure_null_v1")
        self.assertAlmostEqual(problem.metadata["normalization"], np.sqrt(3.5))
        self.assertTrue(np.array_equal(np.diag(problem.Q), np.zeros(8)))

    def test_parameter_validation(self) -> None:
        generators = (
            lambda: weighted_densest_k_subgraph_qubo(1, 0, 0.5, 1),
            lambda: weighted_densest_k_subgraph_qubo(5, 2, 1.1, 1),
            lambda: weighted_densest_k_subgraph_qubo(5, 2, 0.5, 1, weight_range=(-1, 2)),
            lambda: weighted_densest_k_subgraph_qubo(5, 2, 0.5, 1, weight_distribution="gamma"),
            lambda: weighted_max_k_vertex_cover_qubo(5, 5, 0.5, 1),
            lambda: weighted_max_k_vertex_cover_qubo(5, 2, 0.5, 1, hub_strength=0),
            lambda: weak_structure_null_qubo(5, 2, 0.5, 0, 1),
        )
        for generator in generators:
            with self.subTest(generator=generator), self.assertRaises(ValueError):
                generator()


if __name__ == "__main__":
    unittest.main()
