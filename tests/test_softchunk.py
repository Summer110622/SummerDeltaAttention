import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from summer_delta_attention import SummerDeltaAttention


@unittest.skipIf(torch is None, "install the optional trainable extra")
class SoftChunkTests(unittest.TestCase):
    def test_chunk_state_matches_original_sequential_recurrence(self):
        from summer_delta_attention.softchunk import delta_state_chunked
        rng = np.random.default_rng(314)
        for dtype, tolerance in ((np.float32, 3e-5), (np.float64, 1e-11)):
            keys = rng.normal(size=(2, 137, 7)).astype(dtype)
            keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
            values = rng.normal(size=(2, 137, 5)).astype(dtype)
            expected = []
            for batch in range(2):
                memory = SummerDeltaAttention(7, 5, capacity=0, beta=0.05, dtype=dtype)
                for key, value in zip(keys[batch], values[batch]):
                    memory.write(key, value)
                expected.append(memory.state)
            k, v = torch.tensor(keys), torch.tensor(values)
            for chunk in (1, 7, 64, 128, 200):
                actual = delta_state_chunked(k, v, chunk_size=chunk)
                np.testing.assert_allclose(actual.numpy(), expected, atol=tolerance, rtol=tolerance)

    def test_chunk_solve_and_soft_correction_gradchecks(self):
        from summer_delta_attention.softchunk import delta_state_chunked, soft_archive_read
        torch.manual_seed(41)
        k = torch.randn(1, 6, 3, dtype=torch.float64, requires_grad=True)
        v = torch.randn(1, 6, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: delta_state_chunked(a, b, chunk_size=4), (k, v)))
        q = torch.nn.functional.normalize(torch.randn(1, 2, 3, dtype=torch.float64), dim=-1).requires_grad_()
        kn = torch.nn.functional.normalize(torch.randn(1, 6, 3, dtype=torch.float64), dim=-1).requires_grad_()
        vv = torch.randn(1, 6, 2, dtype=torch.float64, requires_grad=True)
        state = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
        scale = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        gate = torch.full((1, 2, 1), 0.7, dtype=torch.float64, requires_grad=True)
        indices = torch.tensor([0, 3, 5])
        f = lambda a, b, c, d, e, g: soft_archive_read(a, b, c, indices, d, scale=e, gate=g)
        self.assertTrue(torch.autograd.gradcheck(f, (q, kn, vv, state, scale, gate)))

    def test_invalid_inputs_and_empty_archive(self):
        from summer_delta_attention.softchunk import SoftChunkMixer, delta_state_chunked
        with self.assertRaises(ValueError):
            delta_state_chunked(torch.randn(2, 3), torch.randn(2, 4))
        with self.assertRaises(ValueError):
            delta_state_chunked(torch.randn(1, 2, 3), torch.randn(1, 2, 4), chunk_size=0)
        layer = SoftChunkMixer(8, capacity=0)
        x = torch.randn(2, 5, 8)
        self.assertEqual(layer(x).shape, (2, 5, 8))
        with self.assertRaises(ValueError):
            layer(x, valid_mask=torch.ones(2, 4, dtype=torch.bool))

    def test_archive_mask_and_padding_invariance(self):
        from summer_delta_attention.softchunk import SoftChunkMixer, soft_archive_read
        torch.manual_seed(42)
        q = torch.nn.functional.normalize(torch.randn(1, 1, 3), dim=-1)
        k = torch.nn.functional.normalize(torch.randn(1, 4, 3), dim=-1)
        v = torch.randn(1, 4, 2)
        state = torch.randn(1, 3, 2)
        gate = torch.full((1, 1, 1), 0.8)
        all_masked = soft_archive_read(q, k, v, torch.tensor([[0, 1, 2, 3]]), state,
                                      gate=gate, archive_mask=torch.zeros(1, 4, dtype=torch.bool))
        torch.testing.assert_close(all_masked, torch.bmm(q, state))

        layer = SoftChunkMixer(8, capacity=4, seed=3).eval()
        tokens = torch.randn(2, 5, 8)
        valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        expected = layer(tokens, valid_mask=valid)[:1]
        padded = torch.cat((tokens[:1, :3], torch.randn(1, 2, 8)), dim=1)
        actual = layer(padded, valid_mask=torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool))
        torch.testing.assert_close(actual[:, :3], expected[:, :3], atol=1e-6, rtol=1e-6)

    def test_learnable_gates_temperature_and_parameter_budget(self):
        from summer_delta_attention.softchunk import SoftChunkMixer
        torch.manual_seed(43)
        mixer = SoftChunkMixer(8, capacity=8)
        loss = mixer(torch.randn(2, 9, 8)).square().mean()
        loss.backward()
        for name, param in mixer.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        self.assertGreater(mixer.log_inverse_temperature.grad.abs().item(), 0)
        self.assertGreater(mixer.gate.weight.grad.abs().sum().item(), 0)
        # Q,K,V,out each d*d; one scalar temperature; gate Wx+b.
        self.assertEqual(sum(p.numel() for p in mixer.parameters()), 4 * 8 * 8 + 8 + 2)


if __name__ == "__main__":
    unittest.main()
