import unittest

import numpy as np

from summer_delta_attention import SummerDeltaAttention


class BatchReadTests(unittest.TestCase):
    def assert_matches_scalar(self, memory, queries, **kwargs):
        expected = [memory.read(q, details=True) for q in queries]
        actual, info = memory.read_batch(queries, details=True, **kwargs)
        atol = 2e-6 if memory.dtype == np.float32 else 1e-12
        np.testing.assert_allclose(actual, [x[0] for x in expected], atol=atol, rtol=atol)
        np.testing.assert_allclose(info["fast"], [x[1]["fast"] for x in expected],
                                   atol=atol, rtol=atol)
        np.testing.assert_allclose(info["similarity"],
                                   [x[1]["similarity"] for x in expected],
                                   atol=atol, rtol=atol, equal_nan=True)
        np.testing.assert_array_equal(info["accepted"],
                                      [x[1]["accepted"] for x in expected])
        self.assertEqual(info["archive_size"], memory.archive_size)
        self.assertEqual(actual.dtype, memory.dtype)
        np.testing.assert_array_equal(actual, memory.read_batch(queries, **kwargs))
        return actual, info

    def test_random_queries_retained_keys_and_reservoir_eviction(self):
        rng = np.random.default_rng(41)
        keys, values = rng.normal(size=(90, 16)), rng.normal(size=(90, 7))
        for dtype in (np.float32, np.float64):
            memory = SummerDeltaAttention(16, 7, capacity=33, dtype=dtype,
                                          threshold=0.65, seed=7)
            for k, v in zip(keys, values):
                memory.write(k, v)
            queries = np.concatenate([memory._keys[:33], rng.normal(size=(19, 16))])
            for batch_size, archive_block_size in ((1, 1), (7, 8), (256, 4096)):
                with self.subTest(dtype=dtype, batch_size=batch_size,
                                  archive_block_size=archive_block_size):
                    _, info = self.assert_matches_scalar(
                        memory, queries, batch_size=batch_size,
                        archive_block_size=archive_block_size)
                    self.assertTrue(info["accepted"][:33].all())
                    self.assertTrue((~info["accepted"][33:]).any())

    def test_empty_archive_with_and_without_fast_state(self):
        for capacity in (0, 8):
            memory = SummerDeltaAttention(3, 2, capacity=capacity)
            if capacity == 0:
                memory.write([1., 0., 0.], [2., 3.])
            self.assert_matches_scalar(memory, np.eye(3), batch_size=2)

    def test_empty_queries(self):
        memory = SummerDeltaAttention(3, 2, dtype=np.float32)
        for populated in (False, True):
            if populated:
                memory.write([1., 0., 0.], [2., 3.])
            result, info = memory.read_batch(np.empty((0, 3)), details=True)
            self.assertEqual(result.shape, (0, 2))
            self.assertEqual(result.dtype, np.float32)
            self.assertEqual(info["fast"].shape, (0, 2))
            self.assertEqual(info["accepted"].shape, (0,))
            self.assertEqual(info["similarity"].shape, (0,))

    def test_huge_tiny_and_noncontiguous_queries(self):
        memory = SummerDeltaAttention(2, 1, threshold=0.9)
        memory.write([1., 1.], [3.])
        queries = np.array([[1e300, 0, 1e300, 0], [1e-300, 0, 1e-300, 0]])[:, ::2]
        self.assertFalse(queries.flags.c_contiguous)
        before = queries.copy()
        self.assert_matches_scalar(memory, queries, batch_size=1)
        np.testing.assert_array_equal(queries, before)

    def test_exact_tie_across_archive_blocks_uses_first_slot(self):
        memory = SummerDeltaAttention(3, 1, threshold=0.0)
        memory.write([1., 0., 0.], [5.])
        memory.write([0., 1., 0.], [-8.])
        self.assert_matches_scalar(memory, np.array([[1., 1., 0.], [0., 0., 1.]]),
                                   archive_block_size=1)

    def test_gate_boundary_and_near_ties_match_scalar_decisions(self):
        rng = np.random.default_rng(42)
        for dtype in (np.float32, np.float64):
            memory = SummerDeltaAttention(32, 3, dtype=dtype, overwrite_tol=0)
            for _ in range(20):
                memory.write(rng.normal(size=32), rng.normal(size=3))
            query = rng.normal(size=32).astype(dtype)
            _, scalar = memory.read(query, details=True)
            boundary = dtype(scalar["similarity"])
            for threshold in (boundary, np.nextafter(boundary, dtype(np.inf)),
                              np.nextafter(boundary, dtype(-np.inf))):
                memory.threshold = float(threshold)
                self.assert_matches_scalar(memory, np.tile(query, (3, 1)),
                                           archive_block_size=7)
            memory.threshold = 1.0
            self.assert_matches_scalar(memory, memory._keys[:memory.archive_size].copy())

    def test_near_ties_and_overwritten_values(self):
        memory = SummerDeltaAttention(3, 1, threshold=0.0, dtype=np.float64)
        memory.write([1., 0., 0.], [2.])
        memory.write([0., 1., 0.], [7.])
        memory.write([1., 0., 0.], [9.])
        queries = np.array([[1., 1. + 1e-15, 0.], [1., 0., 0.]])
        self.assert_matches_scalar(memory, queries, archive_block_size=1)

    def test_no_state_rng_or_output_aliasing(self):
        memory = SummerDeltaAttention(2, 2, capacity=2)
        memory.write([1., 0.], [3., 5.])
        state, keys, values = memory.state.copy(), memory._keys.copy(), memory._values.copy()
        rng = memory._rng.bit_generator.state
        output, info = memory.read_batch(np.eye(2), details=True)
        self.assertFalse(np.shares_memory(output, info["fast"]))
        output[:] = -1
        info["fast"][:] = -2
        np.testing.assert_array_equal(memory.state, state)
        np.testing.assert_array_equal(memory._keys, keys)
        np.testing.assert_array_equal(memory._values, values)
        self.assertEqual(memory._rng.bit_generator.state, rng)

    def test_validation(self):
        memory = SummerDeltaAttention(3, 1)
        for queries in ([], [1., 0., 0.], np.ones((2, 4)), np.ones((2, 1, 3)),
                        [[0., 0., 0.]], [[np.inf, 0., 0.]], [[np.nan, 1., 0.]],
                        [[1., 0., 0.], [0., 0., 0.]]):
            with self.subTest(queries=queries), self.assertRaises(ValueError):
                memory.read_batch(queries, batch_size=1)
        for name in ("batch_size", "archive_block_size"):
            for size in (0, -1, 1.5, True, np.bool_(True), "2"):
                with self.subTest(name=name, size=size), self.assertRaises(ValueError):
                    memory.read_batch(np.eye(3), **{name: size})
        self.assert_matches_scalar(memory, np.eye(3), batch_size=np.int64(2))


if __name__ == "__main__":
    unittest.main()
