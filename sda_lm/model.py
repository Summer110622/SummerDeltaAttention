"""SDA-RC: a small causal LM mixing a frozen reservoir with Summer Delta Attention.

Each block:

    h  = RMSNorm(x)
    r  = Readout(Reservoir(h))            # frozen dynamics, trained linear readout
    x  = x + Mixer(h, cond=r)             # SDA (delta state + gated archive)
    x  = x + SwiGLU(RMSNorm(x))

The reservoir supplies multi-timescale context (what token shift / short conv
does in Mamba-style or DeltaNet-style models) at zero trained recurrent
parameters; SDA supplies precise associative recall. Mixer variants exist so
the same harness can train baselines:

    mixer="sda"        hard-gated archive (the SDA-5 rule, made trainable)
    mixer="sda_soft"   softmax archive instead of hard top-1
    mixer="delta"      DeltaNet only (archive off)
    mixer="attn"       causal softmax self-attention (Transformer baseline)
    mixer="reservoir"  reservoir readout only (echo-state LM baseline)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .reservoir import LinearReservoir
from .sda_layer import SummerDeltaAttentionLayer, reservoir_sample_mask


@dataclass
class SDARCConfig:
    vocab_size: int = 256
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_key: int = 32
    d_value: int = 32
    mixer: str = "sda"
    use_reservoir: bool = True
    n_modes: int = 64
    archive_budget: int | None = None   # None = keep every past token
    tau_init: float = 0.75
    mlp_mult: int = 2
    max_len: int = 1024                 # only used by attn positional embedding
    seed: int = 0


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: int):
        super().__init__()
        self.up = nn.Linear(d, 2 * mult * d, bias=False)
        self.down = nn.Linear(mult * d, d, bias=False)

    def forward(self, x):
        a, b = self.up(x).chunk(2, -1)
        return self.down(F.silu(a) * b)


class CausalAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, d_cond: int = 0):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d + d_cond, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, cond=None, archive_mask=None):
        B, T, D = x.shape
        h = x if cond is None else torch.cat([x, cond], -1)
        q, k, v = self.qkv(h).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if archive_mask is not None:  # same bounded-memory budget as SDA
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=archive_mask[:, None])
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(o.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, cfg: SDARCConfig, layer_idx: int):
        super().__init__()
        d = cfg.d_model
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.reservoir = None
        d_cond = 0
        if cfg.use_reservoir or cfg.mixer == "reservoir":
            self.reservoir = LinearReservoir(d, cfg.n_modes, seed=cfg.seed * 1000 + layer_idx)
            self.readout = nn.Linear(self.reservoir.feature_dim, d)
            d_cond = d
        if cfg.mixer in ("sda", "sda_soft", "delta"):
            archive = {"sda": "hard", "sda_soft": "soft", "delta": "none"}[cfg.mixer]
            self.mixer = SummerDeltaAttentionLayer(
                d, cfg.n_heads, cfg.d_key, cfg.d_value, d_cond=d_cond, archive=archive,
                archive_budget=cfg.archive_budget, tau_init=cfg.tau_init)
        elif cfg.mixer == "attn":
            self.mixer = CausalAttention(d, cfg.n_heads, d_cond)
        elif cfg.mixer == "reservoir":
            self.mixer = None
        else:
            raise ValueError(f"unknown mixer {cfg.mixer}")
        self.mlp = SwiGLU(d, cfg.mlp_mult)

    def forward(self, x, archive_mask=None):
        h = self.n1(x)
        cond = self.readout(self.reservoir(h)) if self.reservoir is not None else None
        if self.mixer is None:
            x = x + cond
        else:
            x = x + self.mixer(h, cond=cond, archive_mask=archive_mask)
        return x + self.mlp(self.n2(x))


class SDARCLM(nn.Module):
    def __init__(self, cfg: SDARCConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Recurrent mixers are order-aware; attention without a reservoir needs positions.
        self.pos = (nn.Embedding(cfg.max_len, cfg.d_model)
                    if cfg.mixer == "attn" and not cfg.use_reservoir else None)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, idx: torch.Tensor, archive_seed: int | None = None):
        B, T = idx.shape
        x = self.emb(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(T, device=idx.device))
        mask = None
        if self.cfg.archive_budget is not None and self.cfg.mixer in ("sda", "sda_soft", "attn"):
            # One reservoir sample per sequence, shared by all layers and heads.
            mask = reservoir_sample_mask(B, T, self.cfg.archive_budget,
                                         seed=archive_seed, device=idx.device)
        for blk in self.blocks:
            x = blk(x, archive_mask=mask)
        return self.head(self.norm(x))

    def param_counts(self) -> dict:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() * (2 if b.is_complex() else 1)
                     for n, b in self.named_buffers() if "reservoir" in n)
        return {"trained": trained, "frozen_reservoir": frozen}
