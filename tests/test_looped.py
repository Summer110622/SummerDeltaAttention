import unittest

import numpy as np

try:
    import torch
    from summer_delta_attention.looped import (
        LoopedSDN, SDALayer, apply_rope, rotary_embedding)
except ImportError:  # torch is an optional dependency
    torch = None

from summer_delta_attention import SummerDeltaAttention


@unittest.skipIf(torch is None, "PyTorch not installed")
class RoPETests(unittest.TestCase):
    def test_preserves_norm(self):
        x = torch.randn(2, 3, 11, 8, dtype=torch.float64)
        cos, sin = rotary_embedding(11, 8, dtype=torch.float64)
        y = apply_rope(x, cos, sin)
        torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))

    def test_dot_product_depends_only_on_relative_position(self):
        D, T = 8, 20
        q, k = torch.randn(D, dtype=torch.float64), torch.randn(D, dtype=torch.float64)
        cos, sin = rotary_embedding(T, D, dtype=torch.float64)
        Q = apply_rope(q.expand(T, D), cos, sin)
        K = apply_rope(k.expand(T, D), cos, sin)
        # score(m, n) for equal offsets m - n must coincide
        a = (Q[5] * K[2]).sum()
        b = (Q[15] * K[12]).sum()
        torch.testing.assert_close(a, b)
        self.assertGreater(abs(float((Q[5] * K[2]).sum() - (Q[5] * K[3]).sum())), 1e-6)

    def test_odd_head_dim_rejected(self):
        with self.assertRaises(ValueError):
            rotary_embedding(4, 7)


@unittest.skipIf(torch is None, "PyTorch not installed")
class SDALayerTests(unittest.TestCase):
    def _layer(self, **kw):
        torch.manual_seed(0)
        return SDALayer(16, 2, **kw).double()

    def test_causal(self):
        layer = self._layer(capacity=4).eval()
        x = torch.randn(2, 9, 16, dtype=torch.float64)
        y1 = layer(x, generator=torch.Generator().manual_seed(1))
        x2 = x.clone()
        x2[:, 6:] = torch.randn_like(x2[:, 6:])
        y2 = layer(x2, generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(y1[:, :6], y2[:, :6])
        self.assertFalse(torch.allclose(y1[:, 6:], y2[:, 6:]))

    def test_matches_numpy_reference_when_archive_holds_everything(self):
        beta, tau, T, H, D = 0.3, 0.5, 8, 2, 8
        layer = self._layer(capacity=T, threshold=tau, beta_max=1.0)
        with torch.no_grad():
            layer.beta_proj.weight.zero_()
            layer.beta_proj.bias.fill_(math.log(beta / (1 - beta)))
            layer.out.weight.copy_(torch.eye(16, dtype=torch.float64))
        x = torch.randn(1, T, 16, dtype=torch.float64)
        got = layer(x)[0].detach().numpy()                       # (T, 16)

        q, k, v = layer.qkv(x).view(1, T, 3, H, D).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        cos, sin = rotary_embedding(T, D, dtype=torch.float64)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        want = np.zeros((T, H, D))
        for h in range(H):
            ref = SummerDeltaAttention(D, D, capacity=T, beta=beta, threshold=tau)
            for t in range(T):
                ref.write(k[0, h, t].detach().numpy(), v[0, h, t].detach().numpy())
                want[t, h] = ref.read(q[0, h, t].detach().numpy())
        np.testing.assert_allclose(got, want.reshape(T, H * D), atol=1e-9)

    def test_archive_is_bounded_and_stats_reported(self):
        layer = self._layer(capacity=3)
        _, stats = layer(torch.randn(1, 20, 16, dtype=torch.float64),
                         return_stats=True)
        self.assertEqual(stats["archive_size"], 3)
        self.assertTrue(0.0 <= stats["accept_rate"] <= 1.0)

    def test_zero_capacity_is_pure_delta_rule(self):
        layer = self._layer(capacity=0)
        y, stats = layer(torch.randn(1, 6, 16, dtype=torch.float64),
                         return_stats=True)
        self.assertEqual(stats["archive_size"], 0)
        self.assertTrue(torch.isfinite(y).all())

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            SDALayer(15, 2)
        with self.assertRaises(ValueError):
            SDALayer(16, 2, beta_max=0.0)


@unittest.skipIf(torch is None, "PyTorch not installed")
class LoopedSDNTests(unittest.TestCase):
    def test_parameter_count_independent_of_loops(self):
        torch.manual_seed(0)
        a = LoopedSDN(50, 32, 2, n_loops=1, max_loops=8)
        b = LoopedSDN(50, 32, 2, n_loops=8, max_loops=8)
        self.assertEqual(sum(p.numel() for p in a.parameters()),
                         sum(p.numel() for p in b.parameters()))

    def test_forward_shapes_and_loop_override(self):
        model = LoopedSDN(50, 32, 2, n_loops=2, max_loops=4, capacity=4).eval()
        tok = torch.randint(0, 50, (2, 10))
        self.assertEqual(model(tok).shape, (2, 10, 50))
        self.assertEqual(model(tok, n_loops=4).shape, (2, 10, 50))
        with self.assertRaises(ValueError):
            model(tok, n_loops=5)

    def test_more_loops_change_the_output(self):
        torch.manual_seed(0)
        model = LoopedSDN(50, 32, 2, n_loops=1, max_loops=3, capacity=4).eval()
        tok = torch.randint(0, 50, (1, 8))
        self.assertFalse(torch.allclose(model(tok, n_loops=1), model(tok, n_loops=3)))

    def test_gradients_reach_every_parameter_and_loss_decreases(self):
        torch.manual_seed(0)
        model = LoopedSDN(16, 32, 2, n_loops=3, capacity=4,
                          gate_temperature=0.1)
        tok = torch.randint(0, 16, (4, 12))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

        def loss_fn():
            logits = model(tok[:, :-1])
            return torch.nn.functional.cross_entropy(
                logits.reshape(-1, 16), tok[:, 1:].reshape(-1))

        first = loss_fn()
        first.backward()
        for name, p in model.named_parameters():
            if name == "loop_emb":      # rows beyond n_loops legitimately get no grad
                continue
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        opt.zero_grad()
        for _ in range(40):
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            opt.step()
        self.assertLess(float(loss.detach()), float(first.detach()) - 0.5)


import math  # noqa: E402  (used by the reference-comparison test)

if __name__ == "__main__":
    unittest.main()
