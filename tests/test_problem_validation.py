from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from struct2circuit.mixers import exchange_profile_statistics
from struct2circuit.problems import (
    CardinalityQUBO, block_correlated_qubo, weak_structure_null_qubo,
    weighted_densest_k_subgraph_qubo, weighted_max_k_vertex_cover_qubo,
)


class ProblemValidationTests(unittest.TestCase):
    def test_python_and_numpy_integer_scalars_are_valid(self):
        for kind in (int, np.int8, np.int32, np.int64, np.uint8, np.uint64):
            with self.subTest(kind=kind):
                problem = CardinalityQUBO(np.eye(4), np.arange(4.), kind(2))
                self.assertIs(type(problem.k), int)
                self.assertEqual(problem.k, 2)
                self.assertEqual(problem.feasible_basis().shape, (6, 4))
                moments = exchange_profile_statistics(problem.Q, problem.c, problem.k)
                self.assertTrue(np.isfinite(moments.rms).all())

    def test_noninteger_cardinalities_fail_at_construction(self):
        for k in (True, False, np.bool_(True), 1.5, 2.0, np.float32(2),
                  np.float64(2), np.array(2), "2", None):
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "integer scalar"):
                CardinalityQUBO(np.eye(4), np.zeros(4), k)

    def test_out_of_range_integer_cardinalities_fail_at_construction(self):
        for k in (-1, 0, 4, 5, np.int64(0), np.uint64(4)):
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "0 < k < n"):
                CardinalityQUBO(np.eye(4), np.zeros(4), k)

    def test_invalid_matrix_shapes_and_small_n_fail_at_construction(self):
        for q in (np.array(1.), np.zeros(3), np.zeros((2, 3)),
                  np.zeros((2, 2, 2))):
            with self.subTest(shape=q.shape), self.assertRaisesRegex(ValueError, "square"):
                CardinalityQUBO(q, np.zeros(2), 1)
        for n in (0, 1):
            with self.subTest(n=n), self.assertRaisesRegex(ValueError, "at least two"):
                CardinalityQUBO(np.zeros((n, n)), np.zeros(n), 1)

    def test_invalid_linear_vector_shapes_fail_at_construction(self):
        for c in (np.array(0.), np.zeros(2), np.zeros(4),
                  np.zeros((3, 1)), np.zeros((1, 3))):
            with self.subTest(shape=c.shape), self.assertRaisesRegex(ValueError, "one-dimensional"):
                CardinalityQUBO(np.eye(3), c, 1)

    def test_nonfinite_matrix_coefficients_fail_at_construction(self):
        for value in (np.nan, np.inf, -np.inf):
            for diagonal in (True, False):
                q = np.eye(3)
                if diagonal:
                    q[0, 0] = value
                else:
                    q[0, 1] = q[1, 0] = value
                with self.subTest(value=value, diagonal=diagonal):
                    with self.assertRaisesRegex(ValueError, "Q.*finite"):
                        CardinalityQUBO(q, np.zeros(3), 1)

    def test_nonfinite_linear_coefficients_fail_at_construction(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "c.*finite"):
                CardinalityQUBO(np.eye(3), np.array([0., value, 1.]), 1)

    def test_absolute_symmetry_contract_and_input_values_are_preserved(self):
        q = np.zeros((3, 3))
        q[0, 1] = 1e-12
        original = q.copy()
        c = np.array([0., 1., 2.])
        problem = CardinalityQUBO(q, c, np.int64(2))
        exchange_profile_statistics(problem.Q, problem.c, problem.k)
        np.testing.assert_array_equal(problem.Q, original)
        np.testing.assert_array_equal(q, original)
        np.testing.assert_array_equal(problem.c, c)
        q[0, 1] = np.nextafter(1e-12, np.inf)
        with self.assertRaisesRegex(ValueError, "symmetric"):
            CardinalityQUBO(q, c, 2)

    def test_existing_generators_remain_constructor_and_descriptor_compatible(self):
        problems = (
            block_correlated_qubo(6, 2, 7),
            weighted_densest_k_subgraph_qubo(6, 2, .5, 7),
            weighted_max_k_vertex_cover_qubo(6, 2, .5, 7),
            weak_structure_null_qubo(6, 2, .5, 1., 7),
        )
        for generated in problems:
            with self.subTest(name=generated.name):
                wrapped = CardinalityQUBO(generated.Q, generated.c, np.int64(generated.k))
                np.testing.assert_array_equal(wrapped.Q, generated.Q)
                np.testing.assert_array_equal(wrapped.c, generated.c)
                exchange_profile_statistics(wrapped.Q, wrapped.c, wrapped.k)


if __name__ == "__main__":
    unittest.main()
