"""Experimental trainable SDA-SoftChunk encoder primitives (optional PyTorch).

This module is not imported by the NumPy package. It provides a chunk-parallel
delta-rule state update and a bounded, differentiable soft KV correction for
research encoders. Hard reservoir selection and optional hard-padding masks
are discrete. This is not a production kernel or pretrained language model.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised on minimal installs
    raise ImportError("SDA-SoftChunk requires the optional 'trainable' extra: pip install -e '.[trainable]'") from exc


def reservoir_indices(length: int, capacity: int, seed: int = 0) -> torch.Tensor:
    """Return Algorithm-R position indices, reproducible for a fixed seed."""
    if length < 0 or capacity < 0:
        raise ValueError("length and capacity must be nonnegative")
    slots = list(range(min(length, capacity)))
    if capacity:
        rng = np.random.default_rng(seed)
        for position in range(capacity, length):
            slot = int(rng.integers(position + 1))
            if slot < capacity:
                slots[slot] = position
    return torch.tensor(slots, dtype=torch.long)


def delta_state_chunked(keys: torch.Tensor, values: torch.Tensor, *,
                        beta: float = 0.05, chunk_size: int = 64) -> torch.Tensor:
    """Apply the sequential delta rule using equivalent triangular solves.

    For each chunk, solve ``(I + beta * tril(K K^T, -1)) U =
    beta * (V - K S_in)`` and update ``S_out = S_in + K^T U``. This avoids
    materializing all per-token states. Q/K normalization remains the caller's
    responsibility. Float evaluation order may differ from a sequential loop.

    Args:
        keys: ``(batch, time, key_dim)``.
        values: ``(batch, time, value_dim)``.
    Returns:
        Final fast-weight state ``(batch, key_dim, value_dim)``.
    """
    if keys.ndim != 3 or values.ndim != 3 or keys.shape[:2] != values.shape[:2]:
        raise ValueError("keys and values must be aligned rank-three tensors")
    if not 0 < beta <= 1:
        raise ValueError("beta must be in (0, 1]")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError("keys and values must be finite")
    state = keys.new_zeros((keys.shape[0], keys.shape[2], values.shape[2]))
    for start in range(0, keys.shape[1], chunk_size):
        key = keys[:, start:start + chunk_size]
        value = values[:, start:start + chunk_size]
        strict_lower = (beta * torch.bmm(key, key.transpose(1, 2))).tril(-1)
        residual = beta * (value - torch.bmm(key, state))
        writes = torch.linalg.solve_triangular(
            strict_lower, residual, upper=False, unitriangular=True
        )
        state = state + torch.bmm(key.transpose(1, 2), writes)
    return state


def soft_archive_read(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                      indices: torch.Tensor, state: torch.Tensor, *, scale: torch.Tensor | float = 4.0,
                      gate: torch.Tensor, archive_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Read final fast state plus a gated softmax correction over retained KV.

    ``queries`` and ``keys`` must be row-normalized. Indices can be shared
    ``(capacity,)`` or per-example ``(batch, capacity)`` reservoir positions.
    Invalid archive slots are masked before softmax; an empty archive returns
    the fast prediction unchanged.
    """
    if queries.ndim != 3 or keys.ndim != 3 or values.ndim != 3:
        raise ValueError("queries, keys, and values must be rank-three tensors")
    if keys.shape[:2] != values.shape[:2] or queries.shape[0] != keys.shape[0]:
        raise ValueError("queries, keys, and values must have aligned batch dimensions")
    if indices.ndim not in (1, 2):
        raise ValueError("indices must have shape (capacity,) or (batch, capacity)")
    fast = torch.bmm(queries, state)
    if indices.numel() == 0:
        return fast
    if indices.ndim == 1:
        archive_keys = keys.index_select(1, indices)
        archive_values = values.index_select(1, indices)
    else:
        if indices.shape[0] != keys.shape[0]:
            raise ValueError("per-example indices must match batch size")
        archive_keys = keys.gather(1, indices.unsqueeze(-1).expand(-1, -1, keys.shape[-1]))
        archive_values = values.gather(1, indices.unsqueeze(-1).expand(-1, -1, values.shape[-1]))
    scores = torch.as_tensor(scale, dtype=queries.dtype, device=queries.device) * torch.bmm(
        queries, archive_keys.transpose(1, 2)
    )
    viable = None
    if archive_mask is not None:
        if archive_mask.shape != archive_keys.shape[:2]:
            raise ValueError("archive_mask must have shape (batch, retained_capacity)")
        viable = archive_mask.any(-1)[:, None, None]
        scores = scores.masked_fill(~archive_mask[:, None, :], -torch.inf)
        scores = torch.where(viable, scores, torch.zeros_like(scores))
    weights = scores.softmax(-1)
    if viable is not None:
        weights = weights * viable
    archive = torch.bmm(weights, archive_values)
    correction = archive - torch.bmm(torch.bmm(weights, archive_keys), state)
    if gate.ndim == 2:
        gate = gate.unsqueeze(-1)
    if gate.shape not in ((queries.shape[0], queries.shape[1], 1),
                          (queries.shape[0], 1, 1)):
        raise ValueError("gate must have shape (batch, queries) or (batch, queries, 1)")
    if viable is not None:
        gate = gate * viable
    return fast + gate * correction


class SoftChunkMixer(nn.Module):
    """Learned projections and a compact SDA-SoftChunk read/write operator.

    The ``forward`` call is a noncausal read-after-write encoder adaptation:
    the complete input is written, then the specified queries read the final
    state/archive. Keys are L2-normalized before recurrence and retrieval.
    """
    def __init__(self, width: int, *, capacity: int = 64, beta: float = 0.05,
                 chunk_size: int = 64, seed: int = 0, gate_init: float = 0.8,
                 inverse_temperature: float = 4.0):
        super().__init__()
        if width < 1 or capacity < 0 or not 0 < gate_init < 1 or inverse_temperature <= 0:
            raise ValueError("invalid width, capacity, or archive initialization")
        self.width, self.capacity, self.beta, self.chunk_size, self.seed = (
            width, capacity, beta, chunk_size, seed
        )
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.log_inverse_temperature = nn.Parameter(torch.tensor(float(np.log(inverse_temperature))))
        self.gate = nn.Linear(width, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(np.log(gate_init / (1.0 - gate_init))))

    def _reservoir(self, valid_mask: torch.Tensor | None, batch: int, length: int,
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
        if valid_mask is None:
            return reservoir_indices(length, self.capacity, self.seed).to(device), None
        if valid_mask.shape != (batch, length):
            raise ValueError("valid_mask must have shape (batch, time)")
        rows, masks = [], []
        width = min(self.capacity, length)
        for valid in valid_mask:
            positions = valid.nonzero().flatten()
            slots = reservoir_indices(len(positions), self.capacity, self.seed).to(device)
            selected = positions.index_select(0, slots)
            rows.append(torch.nn.functional.pad(selected, (0, width - len(selected))))
            masks.append(torch.arange(width, device=device) < len(selected))
        return torch.stack(rows), torch.stack(masks)

    def forward(self, inputs: torch.Tensor, *, valid_mask: torch.Tensor | None = None,
                query_indices: torch.Tensor | None = None) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.width:
            raise ValueError("inputs must have shape (batch, time, width)")
        batch, length = inputs.shape[:2]
        queries_in = inputs if query_indices is None else inputs.index_select(1, query_indices)
        q = torch.nn.functional.normalize(self.q(queries_in), dim=-1)
        k = torch.nn.functional.normalize(self.k(inputs), dim=-1)
        v = self.v(inputs)
        if valid_mask is not None:
            if valid_mask.shape != (batch, length):
                raise ValueError("valid_mask must have shape (batch, time)")
            mask = valid_mask.unsqueeze(-1)
            k = k * mask
            v = v * mask
        state = delta_state_chunked(k, v, beta=self.beta, chunk_size=self.chunk_size)
        indices, archive_mask = self._reservoir(valid_mask, batch, length, inputs.device)
        temperature = self.log_inverse_temperature.exp().clamp(0.25, 32.0)
        gate = self.gate(queries_in).sigmoid()
        mixed = soft_archive_read(q, k, v, indices, state, scale=temperature,
                                  gate=gate, archive_mask=archive_mask)
        return self.out(mixed)
