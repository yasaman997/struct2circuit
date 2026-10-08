from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from struct2circuit.mixers import exchange_profile_scores, score_conditioned_mixer


class ScoreConditionedMixerTests(unittest.TestCase):
    def assert_connected(self, n: int, edges: tuple[tuple[int, int], ...]) -> None:
        # Independent reachability check, rather than the constructor's helper.
        reached = {0}
        for _ in range(n):
            for i, j in edges:
                if i in reached or j in reached:
                    reached.update((i, j))
        self.assertEqual(reached, set(range(n)))

    def test_all_small_integer_budgets_are_exact_and_connected(self) -> None:
        rng = np.random.default_rng(408)
        for n in range(2, 9):
            edges = tuple(combinations(range(n), 2))
            unique_scores = np.zeros((n, n))
            for (i, j), value in zip(edges, rng.permutation(len(edges)), strict=True):
                unique_scores[i, j] = unique_scores[j, i] = value
            for scores in (unique_scores, np.ones((n, n))):
                for budget in range(n - 1, len(edges) + 1):
                    for prefer in ("low", "high"):
                        with self.subTest(n=n, budget=budget, prefer=prefer):
                            mixer = score_conditioned_mixer(scores, budget, prefer=prefer)
                            self.assertEqual(mixer.edge_count, budget)
                            self.assertEqual(len(set(mixer.edges)), budget)
                            self.assert_connected(n, mixer.edges)
                            self.assertEqual(
                                mixer.edges,
                                score_conditioned_mixer(scores, budget, prefer=prefer).edges,
                            )
                            if budget == len(edges):
                                self.assertEqual(mixer.edges, edges)

    def test_low_and_high_tree_then_remaining_edge_semantics(self) -> None:
        scores = np.array([
            [0, 1, 2, 3],
            [1, 0, 4, 5],
            [2, 4, 0, 6],
            [3, 5, 6, 0],
        ], dtype=float)
        self.assertEqual(score_conditioned_mixer(scores, 3, prefer="low").edges, ((0, 1), (0, 2), (0, 3)))
        self.assertEqual(score_conditioned_mixer(scores, 3, prefer="high").edges, ((0, 3), (1, 3), (2, 3)))
        self.assertEqual(
            score_conditioned_mixer(scores, 4, prefer="low").edges,
            ((0, 1), (0, 2), (0, 3), (1, 2)),
        )
        self.assertEqual(
            score_conditioned_mixer(scores, 4, prefer="high").edges,
            ((0, 3), (1, 2), (1, 3), (2, 3)),
        )

    def test_conditional_rms_pipeline_is_permutation_equivariant_without_ties(self) -> None:
        rng = np.random.default_rng(207)
        n, k = 7, 3
        q = rng.normal(size=(n, n))
        q = (q + q.T) / 2.0
        c = rng.normal(size=n)
        scores = exchange_profile_scores(q, c, k)
        values = scores[np.triu_indices(n, k=1)]
        self.assertEqual(len(np.unique(values)), len(values))
        for _ in range(5):
            permutation = rng.permutation(n)
            permuted = exchange_profile_scores(q[np.ix_(permutation, permutation)], c[permutation], k)
            for budget in range(n - 1, n * (n - 1) // 2 + 1):
                for prefer in ("low", "high"):
                    graph = score_conditioned_mixer(scores, budget, prefer=prefer)
                    relabeled = score_conditioned_mixer(permuted, budget, prefer=prefer)
                    restored = tuple(sorted(
                        tuple(sorted((int(permutation[i]), int(permutation[j]))))
                        for i, j in relabeled.edges
                    ))
                    with self.subTest(budget=budget, prefer=prefer):
                        self.assertEqual(graph.edges, restored)

    def test_exact_ties_are_deterministic_but_label_dependent(self) -> None:
        scores = np.ones((4, 4))
        permutation = np.array([1, 0, 2, 3])
        for prefer in ("low", "high"):
            graph = score_conditioned_mixer(scores, 3, prefer=prefer)
            self.assertEqual(graph.edges, ((0, 1), (0, 2), (0, 3)))
            relabeled = score_conditioned_mixer(scores[np.ix_(permutation, permutation)], 3, prefer=prefer)
            restored = tuple(sorted(
                tuple(sorted((int(permutation[i]), int(permutation[j]))))
                for i, j in relabeled.edges
            ))
            self.assertEqual(restored, ((0, 1), (1, 2), (1, 3)))
            self.assertNotEqual(graph.edges, restored)

    def test_positive_objective_scaling_preserves_selected_topology(self) -> None:
        rng = np.random.default_rng(74)
        q = rng.normal(size=(6, 6))
        q = (q + q.T) / 2.0
        c = rng.normal(size=6)
        scores = exchange_profile_scores(q, c, 3)
        for scale in (1e-12, 1.0, 1e12):
            scaled = exchange_profile_scores(scale * q, scale * c, 3)
            for budget in (5, 8, 15):
                for prefer in ("low", "high"):
                    self.assertEqual(
                        score_conditioned_mixer(scores, budget, prefer=prefer).edges,
                        score_conditioned_mixer(scaled, budget, prefer=prefer).edges,
                    )

    def test_invalid_scores_and_budgets_are_rejected(self) -> None:
        scores = np.ones((4, 4))
        for budget in (2, 7, 3.1, True, np.bool_(True)):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                score_conditioned_mixer(scores, budget)
        for invalid in (
            np.zeros((0, 0)), np.zeros((1, 1)), np.zeros((3, 4)),
            np.full((4, 4), np.nan), np.full((4, 4), np.inf),
            np.triu(np.ones((4, 4))),
        ):
            with self.subTest(scores=invalid), self.assertRaises(ValueError):
                score_conditioned_mixer(invalid, 3)
        with self.assertRaises(ValueError):
            score_conditioned_mixer(scores, 3, prefer="unknown")


if __name__ == "__main__":
    unittest.main()
