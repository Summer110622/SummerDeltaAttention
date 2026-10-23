import importlib.util
import unittest
from pathlib import Path

import numpy as np
from summer_delta_attention import SummerDeltaAttention, kda_proxy_state, full_kv_attention

EXPERIMENT = Path(__file__).resolve().parents[1] / "research" / "experiment.py"


class ReferenceTests(unittest.TestCase):
    def test_exact_archived_key_recovers_original_value(self):
        memory = SummerDeltaAttention(3, 2, capacity=3, beta=0.4, threshold=0.9)
        pairs = [(np.array([1., 0., 0.]), np.array([2., -1.])),
                 (np.array([0., 1., 0.]), np.array([5., 3.])),
                 (np.array([0., 0., 1.]), np.array([-2., 4.]))]
        for k, v in pairs:
            memory.write(k, v)
        for k, v in pairs:
            actual, diagnostic = memory.read(k, details=True)
            np.testing.assert_allclose(actual, v, atol=1e-12)
            self.assertTrue(diagnostic["accepted"])

    def test_zero_capacity_is_no_decay_delta_rule(self):
        memory = SummerDeltaAttention(2, 1, capacity=0, beta=0.5)
        memory.write([1., 0.], [4.])
        memory.write([1., 0.], [2.])
        self.assertAlmostEqual(memory.state[0, 0], 2.0)
        self.assertEqual(memory.archive_size, 0)
        self.assertAlmostEqual(memory.read([1., 0.])[0], 2.0)

    def test_reservoir_memory_is_bounded(self):
        # Distinct keys: rewriting an archived key overwrites it in place.
        memory = SummerDeltaAttention(100, 2, capacity=7, seed=42)
        for j in range(100):
            memory.write(np.eye(100)[j], [float(j), 1.])
        self.assertEqual(memory.archive_size, 7)
        self.assertEqual(memory.bytes_of_state_and_archive, (100 * 2 + 7 * 102) * 8)

    def test_archive_is_not_alias_of_user_data(self):
        memory = SummerDeltaAttention(2, 1, capacity=1, beta=1)
        key = np.array([1., 0.])
        value = np.array([7.])
        memory.write(key, value)
        key[0], value[0] = 0, 3
        np.testing.assert_allclose(memory.read([1., 0.]), [7.])

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            SummerDeltaAttention(0, 2)
        with self.assertRaises(ValueError):
            SummerDeltaAttention(2, 2).write([0., 0.], [1., 2.])
        with self.assertRaises(ValueError):
            SummerDeltaAttention(2, 2).read([0., 0.])
        with self.assertRaises(ValueError):
            SummerDeltaAttention(2, 2).write([np.inf, 0.], [1., 2.])

    def test_proxy_and_full_attention_shapes(self):
        keys = np.eye(3)
        values = np.arange(6, dtype=float).reshape(3, 2)
        state = kda_proxy_state(keys, values)
        self.assertEqual(state.shape, (3, 2))
        self.assertEqual(full_kv_attention(keys[0], keys, values).shape, (2,))

    def test_proxy_rejects_unnormalized_keys(self):
        with self.assertRaises(ValueError):
            kda_proxy_state(np.array([[2., 0.]]), np.array([[1.]]))


class OverwriteTests(unittest.TestCase):
    def test_rewritten_key_returns_latest_value(self):
        memory = SummerDeltaAttention(2, 1, capacity=4, beta=0.5, threshold=0.9)
        memory.write([1., 0.], [1.])
        memory.write([1., 0.], [9.])
        self.assertEqual(memory.archive_size, 1)
        np.testing.assert_allclose(memory.read([1., 0.]), [9.], atol=1e-12)

    def test_rewrite_wins_even_when_archive_is_full(self):
        memory = SummerDeltaAttention(3, 1, capacity=2, beta=0.5, threshold=0.9, seed=3)
        memory.write([1., 0., 0.], [1.])
        memory.write([0., 1., 0.], [2.])
        for new_value in (5., 6., 7.):
            memory.write([1., 0., 0.], [new_value])
            np.testing.assert_allclose(memory.read([1., 0., 0.]), [new_value], atol=1e-12)
        self.assertEqual(memory.archive_size, 2)

    def test_rewrite_is_scale_invariant(self):
        memory = SummerDeltaAttention(2, 1, capacity=2, beta=0.5, threshold=0.9)
        memory.write([1., 0.], [1.])
        memory.write([3., 0.], [4.])
        self.assertEqual(memory.archive_size, 1)
        np.testing.assert_allclose(memory.read([5., 0.]), [4.], atol=1e-12)


class GateTests(unittest.TestCase):
    def test_distant_query_is_rejected_and_uses_fast_state(self):
        memory = SummerDeltaAttention(2, 1, capacity=1, beta=1.0, threshold=0.9)
        memory.write([1., 0.], [5.])
        out, info = memory.read([0., 1.], details=True)
        self.assertFalse(info["accepted"])
        np.testing.assert_allclose(out, info["fast"])
        np.testing.assert_allclose(out, [0.])

    def test_threshold_is_inclusive(self):
        memory = SummerDeltaAttention(2, 1, capacity=1, beta=1.0, threshold=1.0)
        memory.write([1., 0.], [5.])
        _, info = memory.read([1., 0.], details=True)
        self.assertTrue(info["accepted"])

    def test_close_but_inexact_query_error_is_bounded_by_state_norm(self):
        rng = np.random.default_rng(0)
        memory = SummerDeltaAttention(8, 3, capacity=1, beta=0.3, threshold=0.5)
        k, v = rng.normal(size=8), rng.normal(size=3)
        memory.write(k, v)
        k_unit = k / np.linalg.norm(k)
        q = k_unit + 0.1 * rng.normal(size=8) / np.sqrt(8)
        q_unit = q / np.linalg.norm(q)
        out, info = memory.read(q, details=True)
        self.assertTrue(info["accepted"])
        bound = np.linalg.norm(memory.state, 2) * np.linalg.norm(q_unit - k_unit)
        self.assertLessEqual(np.linalg.norm(out - v), bound + 1e-12)


class NumericsTests(unittest.TestCase):
    def test_huge_finite_vectors_do_not_overflow(self):
        memory = SummerDeltaAttention(2, 1, capacity=1, beta=1.0, threshold=0.9)
        memory.write([1e200, 1e200], [3.])
        out = memory.read([1e-200, 1e-200])
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_allclose(out, [3.], atol=1e-12)

    def test_float32_dtype_is_preserved(self):
        memory = SummerDeltaAttention(3, 2, capacity=2, beta=0.5, threshold=0.9,
                                      dtype=np.float32)
        memory.write([1., 0., 0.], [2., -1.])
        out = memory.read([1., 0., 0.])
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, [2., -1.], atol=1e-5)

    def test_same_seed_is_deterministic(self):
        def run(seed):
            m = SummerDeltaAttention(40, 1, capacity=5, seed=seed)
            for j in range(40):
                m.write(np.eye(40)[j], [float(j)])
            return m._values[:m.archive_size, 0].copy()
        np.testing.assert_array_equal(run(11), run(11))


class ReservoirTests(unittest.TestCase):
    def test_every_token_is_retained_with_equal_probability(self):
        n, capacity, trials = 40, 8, 600
        counts = np.zeros(n)
        eye = np.eye(n)
        for seed in range(trials):
            m = SummerDeltaAttention(n, 1, capacity=capacity, seed=seed)
            for j in range(n):
                m.write(eye[j], [float(j)])
            counts[m._values[:m.archive_size, 0].astype(int)] += 1
        expected = trials * capacity / n                # 120
        sigma = np.sqrt(trials * (capacity / n) * (1 - capacity / n))  # ~9.8
        self.assertLess(np.max(np.abs(counts - expected)), 6 * sigma)


@unittest.skipUnless(EXPERIMENT.exists(), "research/experiment.py not found")
class ExperimentEquivalenceTests(unittest.TestCase):
    """The benchmark re-implements SDA; it must agree with the packaged class."""

    @classmethod
    def setUpClass(cls):
        try:
            import pandas  # noqa: F401  (experiment.py imports it at module level)
        except ImportError:
            raise unittest.SkipTest("pandas not installed (pip install -e .[research])")
        spec = importlib.util.spec_from_file_location("sda_experiment", EXPERIMENT)
        cls.exp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.exp)

    def test_state_and_gated_readout_match_benchmark(self):
        exp = self.exp
        n, beta, tau = 64, 0.05, 0.75
        keys, values, ids, z, _ = exp.make_data(20261023, n)
        memory = SummerDeltaAttention(exp.D, exp.V, capacity=n, beta=beta, threshold=tau)
        for k, v in zip(keys, values):
            memory.write(k, v)
        state = exp.state_update(keys, values, beta)
        np.testing.assert_allclose(memory.state, state, atol=1e-12)
        self.assertEqual(memory.archive_size, n)
        for sigma in (0.0, 0.4, 1.0):
            q = exp.queries_for(keys, ids, z, sigma)
            expected = exp.predictions(q, keys, values, state, np.arange(n), tau)["SDA tuned"]
            actual = np.stack([memory.read(row) for row in q])
            np.testing.assert_allclose(actual, expected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
