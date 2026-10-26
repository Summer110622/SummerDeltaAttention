"""Fixed (untrained) reservoir front-end for SDA-RC language models.

Reservoir computing keeps the recurrent dynamics *random and frozen* and trains
only the readout. Here the reservoir is a *linear, diagonal, complex* echo-state
network so it can be evaluated in parallel over the sequence with an FFT
convolution (no Python time loop):

    x_t = lambda * x_{t-1} + W_in u_t          (lambda, W_in frozen buffers)

Echo-state property: |lambda| < 1, so the influence of old inputs fades
geometrically. Eigenvalue moduli are spread log-uniformly in
[r_min, r_max] so the reservoir holds memories on many time scales.

Following "next-generation reservoir computing" (Gauthier et al., 2021), the
feature vector handed to the trainable readout contains the linear state plus
cheap fixed quadratic monomials (products of randomly paired channels), which
gives the readout nonlinear features without making the recurrence nonlinear.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class LinearReservoir(nn.Module):
    """Frozen diagonal complex reservoir with NG-RC style quadratic features.

    Input (B, T, d_in) -> features (B, T, feature_dim). No trainable params.
    """

    def __init__(self, d_in: int, n_modes: int = 64, *, r_min: float = 0.5,
                 r_max: float = 0.995, input_scale: float = 1.0,
                 n_quadratic: int | None = None, seed: int = 0):
        super().__init__()
        if not (0 < r_min <= r_max < 1):
            raise ValueError("Echo-state property needs 0 < r_min <= r_max < 1")
        g = torch.Generator().manual_seed(seed)
        radius = torch.exp(torch.linspace(math.log(r_min), math.log(r_max), n_modes))
        theta = torch.rand(n_modes, generator=g) * math.pi  # oscillation freq.
        lam = torch.polar(radius, theta)
        # Normalise each mode's steady-state gain: sum_t |lam|^(2t) = 1/(1-r^2).
        gain = torch.sqrt(1 - radius ** 2)
        w_in = torch.complex(torch.randn(d_in, n_modes, generator=g),
                             torch.randn(d_in, n_modes, generator=g))
        w_in = w_in * (input_scale / math.sqrt(2 * d_in)) * gain
        self.register_buffer("lam", lam)
        self.register_buffer("w_in", w_in)
        n_lin = 2 * n_modes
        n_quadratic = n_lin if n_quadratic is None else n_quadratic
        pairs = torch.randint(0, n_lin, (2, n_quadratic), generator=g)
        self.register_buffer("pairs", pairs)
        self.n_modes = n_modes
        self.feature_dim = n_lin + n_quadratic

    def states(self, u: torch.Tensor) -> torch.Tensor:
        """Complex reservoir states (B, T, n_modes) via causal FFT convolution."""
        T = u.shape[1]
        drive = torch.einsum("btd,dn->btn", u.to(self.w_in.dtype), self.w_in)
        t = torch.arange(T, device=u.device)
        kernel = self.lam[None, :] ** t[:, None]                 # (T, n)
        n_fft = 2 * T
        X = torch.fft.fft(drive, n=n_fft, dim=1) * torch.fft.fft(kernel, n=n_fft, dim=0)
        return torch.fft.ifft(X, dim=1)[:, :T]

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x = self.states(u)
        lin = torch.cat([x.real, x.imag], dim=-1).to(u.dtype)
        quad = lin[..., self.pairs[0]] * lin[..., self.pairs[1]]
        return torch.cat([lin, quad], dim=-1)


def reservoir_reference_loop(res: LinearReservoir, u: torch.Tensor) -> torch.Tensor:
    """Slow sequential recurrence, used only by tests to check the FFT path."""
    B, T, _ = u.shape
    x = torch.zeros(B, res.n_modes, dtype=res.lam.dtype)
    out = []
    for t in range(T):
        x = res.lam * x + u[:, t].to(res.w_in.dtype) @ res.w_in
        out.append(x)
    return torch.stack(out, dim=1)
