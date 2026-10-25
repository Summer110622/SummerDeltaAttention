"""Same-shape A3 baselines for the frozen SDALM-10M stack.

Two mixers, and nothing else:

* ``softmax`` — causal scaled dot-product attention. One score matmul and a
  causal mask. No token loop, no fla.
* ``delta`` — no-decay delta state with fixed β. No archive, no confidence
  gate, and not the Expanded width.

Neither mixer is a third hybrid. Byte counts use the dtype item size.
``accepted_rate`` is not defined here and must not be copied from SDA.
"""
from __future__ import annotations

import torch
from torch import nn

try:
    from .sda_block import _unit, delta_states, nbytes_for_entries
except ImportError:
    from sda_block import _unit, delta_states, nbytes_for_entries


class SoftmaxAttention(nn.Module):
    """Causal softmax attention at the same width as one SDA block.

    State is the sequence's K and V. It grows with time, so it is not the
    recency ring and not a fixed-width delta matrix.
    """

    state_kind = "causal_kv"

    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if n_head < 1:
            raise ValueError("n_head must be positive")
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = self.d_head ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (batch, time, d_model), got {tuple(x.shape)}")
        batch, time, _ = x.shape
        if time < 1:
            raise ValueError("sequence length must be positive")
        qkv = self.qkv(x).view(batch, time, 3, self.n_head, self.d_head)
        query, key, value = (tensor.permute(0, 2, 1, 3) for tensor in qkv.unbind(dim=2))
        scores = torch.einsum("bhtd,bhsd->bhts", query, key) * self.scale
        pos = torch.arange(time, device=x.device)
        causal = pos[None, :] <= pos[:, None]
        weights = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        mixed = torch.einsum("bhts,bhsv->bhtv", weights, value)
        return self.out_proj(mixed.permute(0, 2, 1, 3).reshape(batch, time, -1))

    def state_nbytes(self, batch: int, time: int, dtype: torch.dtype) -> int:
        """Bytes of K and V for this sequence. Not the recency ring."""
        if batch < 1 or time < 1:
            raise ValueError("batch and time must be positive")
        entries = batch * self.n_head * time * self.d_head * 2
        return nbytes_for_entries(entries, dtype)


class NoDecayDelta(nn.Module):
    """No-decay delta readout. Fixed β, no ring, no gate.

    This is the direct delta baseline. It is not Expanded: the state width
    stays ``d_head``, with no lift and no archive.
    """

    state_kind = "delta_state_no_archive"

    def __init__(self, d_model: int, n_head: int, beta: float = 0.05) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if n_head < 1:
            raise ValueError("n_head must be positive")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must lie in (0, 1)")
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("beta", torch.full((n_head,), float(beta)), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (batch, time, d_model), got {tuple(x.shape)}")
        batch, time, _ = x.shape
        if time < 1:
            raise ValueError("sequence length must be positive")
        qkv = self.qkv(x).view(batch, time, 3, self.n_head, self.d_head)
        query, key, value = (tensor.permute(0, 2, 1, 3) for tensor in qkv.unbind(dim=2))
        query = _unit(query)
        key = _unit(key)
        states = delta_states(key, value, self.beta)
        fast = torch.einsum("bhtdv,bhtd->bhtv", states, query)
        return self.out_proj(fast.permute(0, 2, 1, 3).reshape(batch, time, -1))

    def state_nbytes(self, batch: int, time: int, dtype: torch.dtype) -> int:
        """Bytes of one per-step delta state. No archive term. ``time`` unused."""
        if batch < 1 or time < 1:
            raise ValueError("batch and time must be positive")
        entries = batch * self.n_head * self.d_head * self.d_head
        return nbytes_for_entries(entries, dtype)
