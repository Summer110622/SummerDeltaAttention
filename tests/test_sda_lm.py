"""Tests for the PyTorch SDA-RC prototype (skipped when torch is absent)."""
import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestSDALM(unittest.TestCase):
    def test_reservoir_fft_matches_recurrence(self):
        from sda_lm.reservoir import LinearReservoir, reservoir_reference_loop
        res = LinearReservoir(8, 16, seed=3).double()
        res.lam, res.w_in = res.lam.to(torch.complex128), res.w_in.to(torch.complex128)
        u = torch.randn(2, 40, 8, dtype=torch.float64)
        torch.testing.assert_close(res.states(u), reservoir_reference_loop(res, u))

    def test_reservoir_is_frozen(self):
        from sda_lm import SDARCConfig, SDARCLM
        m = SDARCLM(SDARCConfig(d_model=32, n_heads=2, d_key=8, d_value=16))
        names = [n for n, _ in m.named_parameters()]
        self.assertFalse(any("reservoir." in n for n in names))
        self.assertGreater(m.param_counts()["frozen_reservoir"], 0)

    def test_reservoir_mask_matches_algorithm_r(self):
        from sda_lm import reservoir_sample_mask
        mask = reservoir_sample_mask(64, 50, 8, seed=0)
        sizes = mask.sum(-1)
        self.assertTrue(torch.all(sizes == torch.clamp(torch.arange(1, 51), max=8)))
        self.assertTrue(torch.all(mask.triu(1) == 0))
        # Uniformity: at the end each of 50 tokens is retained w.p. 8/50.
        freq = mask[:, -1].float().mean(0)
        big = reservoir_sample_mask(2000, 50, 8, seed=1)[:, -1].float().mean(0)
        self.assertLess(float((big - 8 / 50).abs().max()), 0.04)
        self.assertEqual(freq.shape, (50,))

    def test_causality(self):
        from sda_lm import SDARCConfig, SDARCLM
        for mixer in ("sda", "sda_soft", "delta", "attn", "reservoir"):
            m = SDARCLM(SDARCConfig(vocab_size=32, d_model=32, n_heads=2, d_key=8,
                                    d_value=16, mixer=mixer)).eval()
            x = torch.randint(0, 32, (1, 20))
            y = x.clone(); y[0, 15:] = (y[0, 15:] + 1) % 32
            with torch.no_grad():
                a, b = m(x, archive_seed=0), m(y, archive_seed=0)
            torch.testing.assert_close(a[:, :15], b[:, :15], msg=mixer)

    def test_layer_matches_numpy_reference_rule(self):
        """With a hard gate (tiny temp) and constant beta the layer reproduces
        the reference read rule o = F^T q + 1[cos>=tau](v* - F^T k*)."""
        from sda_lm.sda_layer import SummerDeltaAttentionLayer
        torch.manual_seed(0)
        L = SummerDeltaAttentionLayer(8, 1, 8, 8, archive="hard", gate_temp=1e-6).double()
        with torch.no_grad():
            L.beta.weight.zero_(); L.beta.bias.fill_(np.log(0.5 / 0.5))  # beta = .5
        T = 12
        x = torch.randn(1, T, 8, dtype=torch.float64)
        q, k = L.qk(x).view(1, T, 2, 1, 8).unbind(2)
        q = torch.nn.functional.normalize(q[0, :, 0], dim=-1)
        k = torch.nn.functional.normalize(k[0, :, 0], dim=-1)
        v = L.v(x)[0]
        S = torch.zeros(8, 8, dtype=torch.float64)
        ref = []
        for t in range(T):
            S = S + 0.5 * torch.outer(k[t], v[t] - k[t] @ S)
            sc = k[:t + 1] @ q[t]; j = int(sc.argmax())
            o = q[t] @ S
            if sc[j] >= float(L.tau.detach()):
                o = o + v[j] - k[j] @ S
            ref.append(o)
        ref = torch.stack(ref)
        # Undo GroupNorm + output projection by comparing pre-norm activations.
        captured = {}
        L.norm.register_forward_hook(lambda mod, inp, out: captured.setdefault("x", inp[0]))
        L(x)
        torch.testing.assert_close(captured["x"].detach(), ref)


if __name__ == "__main__":
    unittest.main()
