from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from struct2circuit.problems import CardinalityQUBO  # noqa: E402


class ExchangeCancellationTests(unittest.TestCase):
    def test_qij_cancels_from_single_exchange_cost_difference(self) -> None:
        rng = np.random.default_rng(20261004)
        n = 7
        k = 3
        x = np.zeros(n)
        x[[0, 2, 5]] = 1.0
        i, j = 0, 1

        q = rng.normal(size=(n, n))
        q = (q + q.T) / 2.0
        c = rng.normal(size=n)

        q_alt = q.copy()
        q_alt[i, j] += 5.0
        q_alt[j, i] += 5.0

        x_swapped = x.copy()
        x_swapped[i], x_swapped[j] = x_swapped[j], x_swapped[i]

        before_after = CardinalityQUBO(q, c, k).costs(np.vstack([x, x_swapped]))
        altered_before_after = CardinalityQUBO(q_alt, c, k).costs(
            np.vstack([x, x_swapped])
        )

        self.assertTrue(
            np.allclose(
                before_after[1] - before_after[0],
                altered_before_after[1] - altered_before_after[0],
                atol=1e-12,
            )
        )


if __name__ == "__main__":
    unittest.main()
