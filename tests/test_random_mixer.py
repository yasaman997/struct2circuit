from __future__ import annotations

import inspect
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from struct2circuit.mixers import graph_connected, random_connected_mixer  # noqa: E402


class RandomConnectedMixerTests(unittest.TestCase):
    def test_connected_simple_exact_budget(self) -> None:
        mixer = random_connected_mixer(8, 8, 101)
        self.assertTrue(graph_connected(mixer.n, mixer.edges))
        self.assertEqual(mixer.edge_count, 8)
        self.assertEqual(len(set(mixer.edges)), 8)
        self.assertTrue(all(i < j for i, j in mixer.edges))

    def test_seed_replay_and_variation(self) -> None:
        self.assertEqual(
            random_connected_mixer(8, 8, 7).edges,
            random_connected_mixer(8, 8, 7).edges,
        )
        samples = {random_connected_mixer(8, 8, seed).edges for seed in range(8)}
        self.assertGreater(len(samples), 1)

    def test_boundary_budgets(self) -> None:
        tree = random_connected_mixer(6, 5, 3)
        complete = random_connected_mixer(6, 15, 3)
        self.assertTrue(graph_connected(6, tree.edges))
        self.assertEqual(complete.edge_count, 15)

    def test_invalid_arguments(self) -> None:
        for args in [(1, 0, 1), (5, 3, 1), (5, 11, 1)]:
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "n|edge_budget"):
                random_connected_mixer(*args)
        for attempts in (0, -1, 1.5, True):
            with self.subTest(attempts=attempts), self.assertRaisesRegex(ValueError, "max_attempts"):
                random_connected_mixer(5, 4, 1, max_attempts=attempts)  # type: ignore[arg-type]

    def test_exhausted_attempts_is_informative(self) -> None:
        class DisconnectedRng:
            def choice(self, population: int, *, size: int, replace: bool) -> np.ndarray:
                self.assertions = (population, size, replace)
                return np.asarray([0, 1, 2, 5, 6])

        rng = DisconnectedRng()
        with patch("struct2circuit.mixers.np.random.default_rng", return_value=rng):
            with self.assertRaisesRegex(RuntimeError, "n=6, edge_budget=5, seed=99"):
                random_connected_mixer(6, 5, 99, max_attempts=1)
        self.assertEqual(rng.assertions, (15, 5, False))

    def test_public_signature_has_no_problem_inputs(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(random_connected_mixer).parameters),
            ("n", "edge_budget", "seed", "max_attempts"),
        )


if __name__ == "__main__":
    unittest.main()
