"""Trainable, differentiable Summer Delta Attention layer (multi-head).

This lifts the NumPy reference (``summer_delta_attention/reference.py``) into a
layer whose projections are learned end-to-end:

    S_t   = S_{t-1} + beta_t k_t (v_t - S_{t-1}^T k_t)^T     (no-decay delta rule)
    j_t   = argmax_{s in A_t} cos(q_t, k_s)                    (archive A_t)
    g_t   = sigmoid((cos(q_t, k_j) - tau) / temp)              (soft confidence gate)
    o_t   = S_t^T q_t + g_t (v_j - S_t^T k_j)
          = S_t^T (q_t - g_t k_j) + g_t v_j

The rewrite on the last line means the archive correction costs a single extra
mat-vec inside the recurrence. The archive A_t is a bounded set of past
(key, value) pairs kept by reservoir sampling (Algorithm R), simulated exactly
per sequence as each token's [insert, evict) interval, so train/eval memory
matches what a streaming implementation could keep.

``archive="soft"`` replaces the hard top-1 by a softmax over the archive (the
README found hard gating loses badly to softmax under query noise).
``archive="none"`` gives a plain DeltaNet head (the original "fast state").

Gradients reach q/k/v/beta/tau through the gate and the gathered values; the
top-1 index itself is not differentiated (straight argmax).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def reservoir_sample_mask(batch: int, T: int, budget: int, seed: int | None = None,
                          device=None) -> torch.Tensor:
    """mask[b, t, s] = True iff token s is in a size-``budget`` Algorithm-R
    reservoir at time t (s <= t). Exact simulation, O(batch * T)."""
    if budget >= T:
        return torch.ones(T, T, dtype=torch.bool, device=device).tril().expand(batch, T, T)
    rng = np.random.default_rng(seed)
    evict = np.full((batch, T), T, dtype=np.int64)
    inserted = np.ones((batch, T), dtype=bool)
    for b in range(batch):
        slots = list(range(budget))  # slot -> token index
        for t in range(budget, T):
            j = rng.integers(0, t + 1)
            if j < budget:
                evict[b, slots[j]] = t
                slots[j] = t
            else:
                inserted[b, t] = False
    t_idx = torch.arange(T)
    s_idx = torch.arange(T)
    ev = torch.from_numpy(evict)
    ins = torch.from_numpy(inserted)
    mask = (s_idx[None, None, :] <= t_idx[None, :, None]) \
        & (t_idx[None, :, None] < ev[:, None, :]) & ins[:, None, :]
    return mask.to(device)


class SummerDeltaAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, d_key: int = 32,
                 d_value: int = 32, *, d_cond: int = 0, archive: str = "hard",
                 archive_budget: int | None = None, tau_init: float = 0.75,
                 gate_temp: float = 0.05, soft_scale: float = 16.0):
        super().__init__()
        if archive not in ("hard", "soft", "none"):
            raise ValueError("archive must be 'hard', 'soft' or 'none'")
        self.h, self.dk, self.dv = n_heads, d_key, d_value
        self.archive, self.budget = archive, archive_budget
        self.gate_temp, self.soft_scale = gate_temp, soft_scale
        d_in = d_model + d_cond
        self.qk = nn.Linear(d_in, 2 * n_heads * d_key, bias=False)
        self.v = nn.Linear(d_in, n_heads * d_value, bias=False)
        self.beta = nn.Linear(d_in, n_heads)
        self.tau = nn.Parameter(torch.full((n_heads,), float(tau_init)))
        self.out = nn.Linear(n_heads * d_value, d_model, bias=False)
        self.norm = nn.GroupNorm(n_heads, n_heads * d_value)
        self.last_stats: dict = {}

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None,
                archive_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        h = x if cond is None else torch.cat([x, cond], dim=-1)
        q, k = self.qk(h).view(B, T, 2, self.h, self.dk).unbind(2)
        q = F.normalize(q, dim=-1).transpose(1, 2)               # (B,H,T,dk)
        k = F.normalize(k, dim=-1).transpose(1, 2)
        v = self.v(h).view(B, T, self.h, self.dv).transpose(1, 2)
        beta = torch.sigmoid(self.beta(h)).transpose(1, 2)       # (B,H,T)

        q_eff, add = q, None
        if self.archive != "none":
            if archive_mask is None:
                budget = self.budget if self.budget is not None else T
                archive_mask = reservoir_sample_mask(B, T, budget, device=x.device)
            m = archive_mask[:, None]                            # (B,1,T,T)
            scores = (q @ k.transpose(-1, -2)).masked_fill(~m, -2.0)
            top, idx = scores.max(dim=-1)                        # (B,H,T)
            g = torch.sigmoid((top - self.tau[None, :, None]) / self.gate_temp)
            g = g * m.any(-1).to(g.dtype)
            if self.archive == "hard":
                k_sel = torch.gather(k, 2, idx[..., None].expand(-1, -1, -1, self.dk))
                v_sel = torch.gather(v, 2, idx[..., None].expand(-1, -1, -1, self.dv))
            else:
                w = torch.softmax(self.soft_scale * scores.masked_fill(~m, -1e4), dim=-1)
                k_sel, v_sel = w @ k, w @ v
            q_eff = q - g[..., None] * k_sel
            add = g[..., None] * v_sel
            self.last_stats = {"gate_mean": float(g.mean()), "tau": self.tau.detach().clone()}

        S = x.new_zeros(B, self.h, self.dk, self.dv)
        outs = []
        for t in range(T):
            kt, vt, bt = k[:, :, t], v[:, :, t], beta[:, :, t, None]
            err = vt - torch.einsum("bhk,bhkv->bhv", kt, S)
            S = S + torch.einsum("bhk,bhv->bhkv", kt, bt * err)
            outs.append(torch.einsum("bhk,bhkv->bhv", q_eff[:, :, t], S))
        o = torch.stack(outs, dim=2)                             # (B,H,T,dv)
        if add is not None:
            o = o + add
        o = o.transpose(1, 2).reshape(B, T, self.h * self.dv)
        o = self.norm(o.transpose(1, 2)).transpose(1, 2)
        return self.out(o)
