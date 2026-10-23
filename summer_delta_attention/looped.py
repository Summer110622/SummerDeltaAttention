"""LoopedSDN + RoPE: a weight-shared, recurrent-depth prototype built on SDA-5.

Research prototype, NOT a trained model and NOT an optimized kernel.

Components
----------
* ``rotary_embedding`` / ``apply_rope``  -- standard rotary position embedding.
* ``SDALayer``    -- batched, multi-head, causal SDA-5 (no-decay delta state +
                     bounded reservoir archive + confidence-gated residual
                     correction). RoPE is applied to q and k *before* the L2
                     normalisation, so the archive cosine gate is a function of
                     relative position as well as content.
* ``LoopedSDNBlock`` -- pre-norm SDA layer + SwiGLU MLP.
* ``LoopedSDN``   -- one block applied ``n_loops`` times with shared weights,
                     input re-injection and a learned per-loop embedding.

Cost caveats (same as the NumPy reference): the token loop is sequential in
Python, and each read scans the archive, O(B * d_key) per token per head per
loop. Total depth compute is ``n_loops`` times that of a single block; the
parameter count does not grow with ``n_loops``.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# RoPE
# --------------------------------------------------------------------------- #
def rotary_embedding(seq_len: int, head_dim: int, *, base: float = 10000.0,
                     offset: int = 0, device=None, dtype=torch.float32
                     ) -> tuple[Tensor, Tensor]:
    """Return (cos, sin) tables of shape (seq_len, head_dim // 2)."""
    if head_dim % 2:
        raise ValueError("RoPE needs an even head dimension")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device,
                                            dtype=torch.float32) / head_dim))
    pos = torch.arange(offset, offset + seq_len, device=device,
                       dtype=torch.float32)
    angles = torch.outer(pos, inv_freq)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate channel pairs (x[..., 2i], x[..., 2i+1]).

    x: (..., T, D); cos/sin: (T, D // 2). The map is orthogonal, so it
    preserves vector norms.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


# --------------------------------------------------------------------------- #
# SDA layer
# --------------------------------------------------------------------------- #
class SDALayer(nn.Module):
    """Causal multi-head SDA-5 with RoPE.

    Per head and token t (k, q are RoPE-rotated then L2-normalised):
        error = v - F^T k
        F     = F + beta_t * k (x) error                  (no decay)
        archive <- reservoir sample of (k, v) pairs, capacity ``capacity``
        j     = argmax_i cos(q, archive_key[i])
        out   = F^T q + 1[cos >= tau] * (archive_value[j] - F^T archive_key[j])

    ``beta_t`` is data-dependent: ``beta_max * sigmoid(w_beta x_t)``.
    ``gate_temperature > 0`` replaces the hard gate by a sigmoid (useful for
    training); the hard gate is non-differentiable w.r.t. the similarity.
    Reservoir decisions are shared across batch and heads (one RNG draw per
    token) so the loop stays vectorised.
    """

    def __init__(self, d_model: int, n_heads: int, *, capacity: int = 32,
                 threshold: float = 0.75, beta_max: float = 1.0,
                 gate_temperature: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.head_dim = d_model // n_heads
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if capacity < 0 or not (0 < beta_max <= 1) or not (-1 <= threshold <= 1):
            raise ValueError("Require capacity >= 0, 0 < beta_max <= 1, "
                             "-1 <= threshold <= 1")
        self.d_model, self.n_heads = d_model, n_heads
        self.capacity, self.threshold = capacity, threshold
        self.beta_max, self.gate_temperature = beta_max, gate_temperature
        self.rope_base = rope_base
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, *, generator: torch.Generator | None = None,
                return_stats: bool = False):
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim
        q, k, v = self.qkv(x).view(B, T, 3, H, D).unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))        # (B, H, T, D)
        cos, sin = rotary_embedding(T, D, base=self.rope_base,
                                    device=x.device, dtype=x.dtype)
        q = F.normalize(apply_rope(q, cos, sin), dim=-1)
        k = F.normalize(apply_rope(k, cos, sin), dim=-1)
        beta = self.beta_max * torch.sigmoid(self.beta_proj(x))  # (B, T, H)
        beta = beta.transpose(1, 2)                              # (B, H, T)

        state = x.new_zeros(B, H, D, D)
        arch_k: list[Tensor] = []
        arch_v: list[Tensor] = []
        outs, accepted_count = [], x.new_zeros(())
        for t in range(T):
            kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]      # (B, H, D)
            err = vt - torch.einsum("bhk,bhkv->bhv", kt, state)
            state = state + beta[:, :, t, None, None] * (
                kt.unsqueeze(-1) * err.unsqueeze(-2))
            if self.capacity > 0:
                if len(arch_k) < self.capacity:
                    arch_k.append(kt); arch_v.append(vt)
                else:
                    j = int(torch.randint(0, t + 1, (1,),
                                          generator=generator).item())
                    if j < self.capacity:
                        arch_k[j], arch_v[j] = kt, vt
            fast = torch.einsum("bhk,bhkv->bhv", qt, state)
            if arch_k:
                keys = torch.stack(arch_k, dim=2)                # (B,H,N,D)
                vals = torch.stack(arch_v, dim=2)
                scores = torch.einsum("bhnd,bhd->bhn", keys, qt)
                top, idx = scores.max(dim=-1)                    # (B, H)
                gi = idx[..., None, None]
                kj = keys.gather(2, gi.expand(-1, -1, 1, D)).squeeze(2)
                vj = vals.gather(2, gi.expand(-1, -1, 1, D)).squeeze(2)
                resid = vj - torch.einsum("bhk,bhkv->bhv", kj, state)
                if self.gate_temperature > 0:
                    gate = torch.sigmoid((top - self.threshold)
                                         / self.gate_temperature)
                else:
                    gate = (top >= self.threshold).to(x.dtype)
                accepted_count = accepted_count + gate.detach().sum()
                fast = fast + gate.unsqueeze(-1) * resid
            outs.append(fast)
        o = torch.stack(outs, dim=2)                             # (B,H,T,D)
        o = o.transpose(1, 2).reshape(B, T, H * D)
        o = self.out(o)
        if return_stats:
            return o, {"accept_rate": float(accepted_count / (B * H * T)),
                       "archive_size": len(arch_k)}
        return o


# --------------------------------------------------------------------------- #
# Looped network
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) \
            * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class LoopedSDNBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, *, mlp_ratio: float = 8 / 3,
                 **sda_kwargs):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d_model), RMSNorm(d_model)
        self.attn = SDALayer(d_model, n_heads, **sda_kwargs)
        self.mlp = SwiGLU(d_model, int(mlp_ratio * d_model))

    def forward(self, x: Tensor, **kw):
        h = self.attn(self.norm1(x), **kw)
        stats = None
        if isinstance(h, tuple):
            h, stats = h
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return (x, stats) if stats is not None else x


class LoopedSDN(nn.Module):
    """Token embedding -> [shared SDA block] x n_loops -> tied LM head.

    h_0 = E(tokens)
    h_{l+1} = Block(h_l + inject * E(tokens) + loop_emb[l])
    Positions come only from RoPE inside the attention layer; there is no
    absolute position embedding.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_loops: int = 4, *, max_loops: int | None = None,
                 inject: bool = True, **sda_kwargs):
        super().__init__()
        self.n_loops = n_loops
        self.max_loops = max_loops or n_loops
        self.inject = inject
        self.embed = nn.Embedding(vocab_size, d_model)
        self.loop_emb = nn.Parameter(torch.zeros(self.max_loops, d_model))
        self.block = LoopedSDNBlock(d_model, n_heads, **sda_kwargs)
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight                      # tied
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.loop_emb, std=0.02)

    def forward(self, tokens: Tensor, *, n_loops: int | None = None,
                seed: int = 0, return_stats: bool = False):
        n = n_loops or self.n_loops
        if n > self.max_loops:
            raise ValueError(f"n_loops={n} exceeds max_loops={self.max_loops}")
        e = self.embed(tokens)
        h, stats = e, []
        for l in range(n):
            gen = torch.Generator().manual_seed(seed + l)
            inp = h + self.loop_emb[l]
            if self.inject:
                inp = inp + e
            res = self.block(inp, generator=gen, return_stats=return_stats)
            if return_stats:
                h, s = res
                stats.append(s)
            else:
                h = res
        logits = self.head(self.norm(h))
        return (logits, stats) if return_stats else logits
