import unittest
import numpy as np
from summer_delta_attention import SummerDeltaAttention, kda_proxy_state, full_kv_attention


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
        memory = SummerDeltaAttention(3, 2, capacity=7, seed=42)
        for j in range(100):
            memory.write([1., float(j % 5), 0.5], [float(j), 1.])
        self.assertEqual(memory.archive_size, 7)
        self.assertEqual(memory.bytes_of_state_and_archive, (3 * 2 + 7 * 5) * 8)

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

    def test_proxy_and_full_attention_shapes(self):
        keys = np.eye(3)
        values = np.arange(6, dtype=float).reshape(3, 2)
        state = kda_proxy_state(keys, values)
        self.assertEqual(state.shape, (3, 2))
        self.assertEqual(full_kv_attention(keys[0], keys, values).shape, (2,))


if __name__ == "__main__":
    unittest.main()
