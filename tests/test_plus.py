import unittest

import numpy as np
from summer_delta_attention import SummerDeltaAttentionPlus as Plus


def _unit_rows(rng, n, d):
    x = rng.normal(size=(n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class PlusTests(unittest.TestCase):
    def test_exact_archived_key_recovers_value_up_to_quantization(self):
        rng = np.random.default_rng(0)
        keys, values = _unit_rows(rng, 8, 32), rng.normal(size=(8, 4))
        for bits, atol in [(None, 1e-9), (8, 0.05), (4, 0.6)]:
            with self.subTest(bits=bits):
                m = Plus(32, 4, capacity=8, bits=bits, gated=False)
                for k, v in zip(keys, values):
                    m.write(k, v)
                for k, v in zip(keys, values):
                    np.testing.assert_allclose(m.read(k), v, atol=atol * max(1, np.abs(values).max()))

    def test_archive_is_bounded_and_accounting_matches(self):
        m = Plus(100, 2, capacity=7, bits=8, seed=3)
        for j in range(100):
            m.write(np.eye(100)[j], [float(j), 1.0])
        self.assertEqual(m.archive_size, 7)
        self.assertEqual(m.bytes_of_state_and_archive, 100 * 2 * 8 + 7 * (102 * 1 + 4))

    def test_fewer_bits_means_fewer_bytes(self):
        sizes = {}
        for bits in (8, 4):
            m = Plus(16, 16, capacity=20, bits=bits)
            for k in _unit_rows(np.random.default_rng(1), 20, 16):
                m.write(k, k)
            sizes[bits] = m.bytes_of_state_and_archive
        self.assertLess(sizes[4], sizes[8])

    def test_capacity_for_budget_fits_and_is_maximal(self):
        for bits in (None, 8, 4):
            budget = 300 * 1024
            cap = Plus.capacity_for_budget(budget, 64, 64, bits=bits, dtype=np.float32)
            per = Plus.entry_bytes(64, 64, bits, np.float32)

            def total(c):
                extra = (max(1, int(c ** 0.5)) * 64 * 4 + c * 4) if c >= 64 else 0
                return 64 * 64 * 4 + c * per + extra
            self.assertLessEqual(total(cap), budget)
            self.assertGreater(total(cap + 1), budget)
        self.assertGreater(Plus.capacity_for_budget(budget, 64, 64, bits=4),
                           Plus.capacity_for_budget(budget, 64, 64, bits=8))

    def test_zero_capacity_is_no_decay_delta_rule(self):
        m = Plus(2, 1, capacity=0, beta=0.5)
        m.write([1., 0.], [4.])
        m.write([1., 0.], [2.])
        self.assertAlmostEqual(m.state[0, 0], 2.0)
        self.assertAlmostEqual(m.read([1., 0.])[0], 2.0)

    def test_use_state_false_keeps_state_zero(self):
        m = Plus(4, 4, capacity=4, use_state=False)
        m.write([1., 0, 0, 0], [1., 2., 3., 4.])
        self.assertFalse(m.state.any())

    def test_rewriting_an_archived_key_overwrites(self):
        m = Plus(3, 1, capacity=4, bits=None, gated=False)
        m.write([1., 0., 0.], [1.])
        m.write([1., 0., 0.], [9.])
        self.assertEqual(m.archive_size, 1)
        self.assertAlmostEqual(m.read([1., 0., 0.])[0], 9.0, places=9)

    def test_full_probe_matches_exact_scan(self):
        rng = np.random.default_rng(5)
        keys, values = _unit_rows(rng, 300, 16), rng.normal(size=(300, 8))
        clustered = Plus(16, 8, capacity=300, n_clusters=6, probe=6, seed=1)
        exact = Plus(16, 8, capacity=300, n_clusters=1, probe=1, seed=1)
        for k, v in zip(keys, values):
            clustered.write(k, v)
            exact.write(k, v)
        for q in _unit_rows(rng, 10, 16):
            np.testing.assert_allclose(clustered.read(q), exact.read(q), atol=1e-10)

    def test_clustered_read_scans_fewer_candidates(self):
        rng = np.random.default_rng(6)
        centers = _unit_rows(rng, 8, 16)
        keys = centers[rng.integers(0, 8, 400)] + 0.1 * rng.normal(size=(400, 16))
        m = Plus(16, 4, capacity=400, n_clusters=16, probe=2, seed=2)
        for k in keys:
            m.write(k, rng.normal(size=4))
        _, info = m.read(keys[0], details=True)
        self.assertLess(info["candidates"], 400)
        self.assertGreater(m.bytes_of_index, 0)

    def test_soft_gate_falls_back_to_fast_state_for_far_queries(self):
        m = Plus(8, 2, capacity=4, threshold=0.5, gate_temperature=0.05, beta=0.5)
        m.write(np.eye(8)[0], [3., -3.])
        far = np.eye(8)[5]
        out, info = m.read(far, details=True)
        self.assertLess(info["gate"], 1e-3)
        np.testing.assert_allclose(out, info["fast"], atol=1e-2)
        _, near = m.read(np.eye(8)[0], details=True)
        self.assertGreater(near["gate"], 0.999)

    def test_ungated_always_returns_archive_answer(self):
        m = Plus(4, 1, capacity=4, bits=None, gated=False)
        m.write([1., 0, 0, 0], [7.])
        _, info = m.read([0., 1., 0., 0.], details=True)
        self.assertEqual(info["gate"], 1.0)

    def test_empty_archive_reads_fast_state(self):
        m = Plus(4, 3, capacity=0)
        np.testing.assert_array_equal(m.read([1., 0, 0, 0]), np.zeros(3))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            Plus(4, 4, bits=9)
        with self.assertRaises(ValueError):
            Plus(4, 4, beta=0.0)
        m = Plus(4, 4)
        with self.assertRaises(ValueError):
            m.write([0., 0, 0, 0], [1., 1, 1, 1])
        with self.assertRaises(ValueError):
            m.read([1., 0, 0])


if __name__ == "__main__":
    unittest.main()
