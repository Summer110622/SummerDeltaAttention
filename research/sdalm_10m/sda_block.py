"""Differentiable Summer Delta Attention block for the frozen SDALM-10M shape.

Two exclusive configs:

* ``recency`` (default) — no-decay delta state, a causal ring of the last
  ``archive_cap`` writes, and a soft gate while training. A hard gate is used
  only when ``hard_gate_eval`` is set and the module is in eval mode.
* ``expanded`` — no ring and no gate. The same entry budget sits in a wider
  no-decay delta state via a fixed, unlearned lift.

The hot path does not iterate tokens in Python. The delta recurrence is an
associative scan (a loop over doubling offsets, log2 of the length) and the
ring is a banded score tensor. That is not a fused kernel and not fla.

Learnable SDA knobs are β, τ, and the gate temperature. Archive ρ is not a
parameter. Memory counts use ``dtype`` item size, never a hardcoded 8.
``accepted_rate`` is computed only for a recency block from that block's own
scores. Expanded does not receive a recency rate, and this block does not
emit ``hit_rate``.
"""
from __future__ import annotations

import math

import torch
from torch import nn


VARIANTS = ("recency", "expanded")


def recency_state_entries(d_head: int, archive_cap: int) -> int:
    """Scalar slots per head: one fast state plus one recency ring."""
    return d_head * d_head + 2 * archive_cap * d_head


def expanded_state_entries(d_head: int, expand_factor: int) -> int:
    """Scalar slots per head when the ring budget sits in a wider state."""
    return expand_factor * d_head * d_head


def resolve_expand_factor(d_head: int, archive_cap: int, override: int | None = None) -> int:
    """Smallest width whose state covers the recency entry budget.

    Recency stores ``d_head**2 + 2 * archive_cap * d_head`` numbers per head.
    Expanded stores ``f * d_head**2``. Default ``f`` is the smallest integer
    with expanded entries >= recency entries. It is a width, not a third rule.
    """
    if d_head < 1 or archive_cap < 1:
        raise ValueError("d_head and archive_cap must be positive")
    if override is not None:
        if override < 1:
            raise ValueError("expand_factor must be >= 1")
        return override
    need = d_head + 2 * archive_cap
    return max(1, (need + d_head - 1) // d_head)


def nbytes_for_entries(n_entries: int, dtype: torch.dtype) -> int:
    """Byte count for ``n_entries`` values of ``dtype``. Uses item size."""
    if n_entries < 0:
        raise ValueError("n_entries must be nonnegative")
    return int(n_entries) * torch.empty((), dtype=dtype).element_size()


def _inv_sigmoid(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("beta_init must lie in (0, 1)")
    return math.log(p / (1.0 - p))


def _atanh(p: float) -> float:
    # math.atanh is 3.11+. The repo still supports 3.10.
    if not -1.0 < p < 1.0:
        raise ValueError("tau_init must lie in (-1, 1)")
    return 0.5 * math.log((1.0 + p) / (1.0 - p))


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _lift(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Fixed channel lift along the last dimension. ``factor=1`` returns ``x``.

    Copy ``i`` is ``(-1)**i * roll(x, i)``. The loop is over the lift width,
    not over tokens. The map is not learned and not an archive.
    """
    if factor == 1:
        return x
    parts = []
    for i in range(factor):
        if i == 0:
            parts.append(x)
        else:
            sign = 1.0 if i % 2 == 0 else -1.0
            parts.append(sign * torch.roll(x, shifts=i, dims=-1))
    return torch.cat(parts, dim=-1)


def causal_band(time: int, cap: int, device: torch.device) -> torch.Tensor:
    """True where source index ``s`` is inside the live ring at query ``t``.

    The ring is the ``cap`` tokens ending at ``t``, including ``t``.
    Shape ``(time, time)``.
    """
    if time < 1 or cap < 1:
        raise ValueError("time and cap must be positive")
    pos = torch.arange(time, device=device)
    row = pos[:, None]
    col = pos[None, :]
    return (col <= row) & ((row - col) < cap)


def delta_states(key: torch.Tensor, value: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Inclusive no-decay delta states for every time index.

    ``key`` is ``(B, H, T, Dk)``, ``value`` is ``(B, H, T, Dv)``, ``beta`` is
    ``(H,)``. Returns ``S`` of shape ``(B, H, T, Dk, Dv)`` with

        S_t = (I - β k_t k_t^T) S_{t-1} + β k_t v_t^T,  S_{-1} = 0.

    Composition of those affine maps is associative, so the prefix is a
    Hillis–Steele scan: each Python iteration doubles the offset and applies
    one batched matmul across the whole sequence. There is no per-token loop.
    """
    batch, heads, time, d_key = key.shape
    d_value = value.shape[-1]
    eye = torch.eye(d_key, dtype=key.dtype, device=key.device)
    outer_k = torch.einsum("bhtd,bhte->bhtde", key, key)
    coeff = beta.view(1, heads, 1, 1, 1)
    transform = eye.view(1, 1, 1, d_key, d_key) - coeff * outer_k
    update = coeff * torch.einsum("bhtd,bhtv->bhtdv", key, value)
    offset = 1
    while offset < time:
        left_t = transform[:, :, :-offset]
        right_t = transform[:, :, offset:]
        left_u = update[:, :, :-offset]
        right_u = update[:, :, offset:]
        transform = torch.cat((transform[:, :, :offset], right_t @ left_t), dim=2)
        update = torch.cat((update[:, :, :offset], right_t @ left_u + right_u), dim=2)
        offset *= 2
    if update.shape[:3] != (batch, heads, time):
        raise RuntimeError("delta scan changed the leading shape")
    return update


def _delta_states_reference(key: torch.Tensor, value: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Token loop used only to check the scan. Not called from ``forward``."""
    batch, heads, time, d_key = key.shape
    d_value = value.shape[-1]
    state = key.new_zeros(batch, heads, d_key, d_value)
    steps = []
    for index in range(time):
        pred = torch.einsum("bhkd,bhk->bhd", state, key[:, :, index])
        err = value[:, :, index] - pred
        state = state + beta[None, :, None, None] * key[:, :, index, :, None] * err[:, :, None, :]
        steps.append(state)
    return torch.stack(steps, dim=2)


def scan_matches_reference(atol: float = 1e-4) -> bool:
    """Finite check that the scan matches the sequential delta rule, with a β gradient."""
    generator = torch.Generator().manual_seed(0)
    key = _unit(torch.randn(2, 2, 9, 4, generator=generator))
    value = torch.randn(2, 2, 9, 3, generator=generator)
    beta = torch.tensor([0.05, 0.2], requires_grad=True)
    got = delta_states(key, value, beta)
    ref = _delta_states_reference(key, value, beta.detach())
    if not torch.allclose(got, ref, atol=atol, rtol=1e-4):
        return False
    got.sum().backward()
    grad = beta.grad
    return grad is not None and bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0.0


def _soft_correct(states: torch.Tensor, fast: torch.Tensor, scores: torch.Tensor,
                  key: torch.Tensor, value: torch.Tensor, tau: torch.Tensor,
                  temp: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    logits = scores.masked_fill(~valid, -1e9) / temp.view(1, -1, 1, 1)
    alpha = torch.softmax(logits, dim=-1)
    cos_exp = (alpha * scores.masked_fill(~valid, 0.0)).sum(dim=-1)
    key_star = torch.einsum("bhts,bhsd->bhtd", alpha, key)
    value_star = torch.einsum("bhts,bhsv->bhtv", alpha, value)
    fast_at_key = torch.einsum("bhtdv,bhtd->bhtv", states, key_star)
    gate = torch.sigmoid((cos_exp - tau.view(1, -1, 1)) / temp.view(1, -1, 1))
    return fast + gate.unsqueeze(-1) * (value_star - fast_at_key)


def _hard_correct(states: torch.Tensor, fast: torch.Tensor, scores: torch.Tensor,
                  key: torch.Tensor, value: torch.Tensor, tau: torch.Tensor,
                  valid: torch.Tensor) -> torch.Tensor:
    masked = scores.masked_fill(~valid, -1e9)
    best, index = masked.max(dim=-1)
    d_key = key.shape[-1]
    d_value = value.shape[-1]
    key_star = torch.gather(key, 2, index.unsqueeze(-1).expand(-1, -1, -1, d_key))
    value_star = torch.gather(value, 2, index.unsqueeze(-1).expand(-1, -1, -1, d_value))
    fast_at_key = torch.einsum("bhtdv,bhtd->bhtv", states, key_star)
    accept = (best >= tau.view(1, -1, 1)).to(fast.dtype).unsqueeze(-1)
    return fast + accept * (value_star - fast_at_key)


class SDABlock(nn.Module):
    """Projections plus exactly one memory layout.

    Training readout (recency), with ``S`` the delta state after the write::

        fast = S^T q
        α    = softmax(cos(q, K_ring) / temperature)
        g    = sigmoid((α · cos - τ) / temperature)
        out  = fast + g * (α V_ring - S^T (α K_ring))

    Eval with ``hard_gate_eval`` replaces ``α`` and ``g`` by argmax and
    ``1[max cos >= τ]``. Expanded never builds a ring.
    """

    def __init__(self, d_model: int, n_head: int, archive_cap: int, variant: str,
                 beta_init: float, tau_init: float, gate_temp_init: float,
                 expand_factor: int | None = None, hard_gate_eval: bool = False) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if archive_cap < 1:
            raise ValueError("archive_cap must be >= 1")
        if not -1.0 < tau_init < 1.0:
            raise ValueError("tau_init must lie in (-1, 1)")
        if gate_temp_init <= 0:
            raise ValueError("gate_temp_init must be positive")
        self.variant = variant
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.hard_gate_eval = bool(hard_gate_eval)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.beta_logit = nn.Parameter(torch.full((n_head,), _inv_sigmoid(beta_init)))
        if variant == "recency":
            if expand_factor is not None:
                raise ValueError("expand_factor is only valid when variant='expanded'")
            self.archive_cap = archive_cap
            self.expand_factor = None
            self.tau_logit = nn.Parameter(torch.full((n_head,), _atanh(tau_init)))
            self.gate_log_temp = nn.Parameter(torch.full((n_head,), math.log(gate_temp_init)))
        else:
            factor = resolve_expand_factor(self.d_head, archive_cap, expand_factor)
            if expanded_state_entries(self.d_head, factor) < recency_state_entries(self.d_head, archive_cap):
                raise ValueError(
                    "expand_factor does not cover the recency state+archive entry budget"
                )
            self.archive_cap = 0
            self.expand_factor = factor

    def rule_state_entries_per_head(self) -> int:
        if self.variant == "recency":
            return recency_state_entries(self.d_head, self.archive_cap)
        return expanded_state_entries(self.d_head, int(self.expand_factor))

    def rule_state_nbytes(self, batch: int, dtype: torch.dtype) -> int:
        """Bytes of one per-step state for this block, not scan workspace."""
        if batch < 1:
            raise ValueError("batch must be positive")
        entries = batch * self.n_head * self.rule_state_entries_per_head()
        return nbytes_for_entries(entries, dtype)

    def state_nbytes(self, batch: int, time: int, dtype: torch.dtype) -> int:
        """Per-step rule bytes. ``time`` is accepted so every mixer shares the call."""
        if time < 1:
            raise ValueError("time must be positive")
        return self.rule_state_nbytes(batch, dtype)

    @property
    def state_kind(self) -> str:
        if self.variant == "recency":
            return "delta_plus_recency_ring"
        return "expanded_delta"

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
        if self.variant == "recency":
            mixed = self._recency(query, key, value)
        else:
            mixed = self._expanded(query, key, value)
        return self.out_proj(mixed.permute(0, 2, 1, 3).reshape(batch, time, -1))

    def _recency(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        beta = torch.sigmoid(self.beta_logit)
        tau = torch.tanh(self.tau_logit)
        states = delta_states(key, value, beta)
        fast = torch.einsum("bhtdv,bhtd->bhtv", states, query)
        scores = torch.einsum("bhtd,bhsd->bhts", query, key)
        valid = causal_band(query.shape[2], self.archive_cap, query.device)
        if self.hard_gate_eval and not self.training:
            return _hard_correct(states, fast, scores, key, value, tau, valid)
        temp = self.gate_log_temp.exp().clamp(min=1e-3, max=10.0)
        return _soft_correct(states, fast, scores, key, value, tau, temp, valid)

    def _expanded(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        factor = int(self.expand_factor)
        key_l = _unit(_lift(key, factor))
        query_l = _unit(_lift(query, factor))
        beta = torch.sigmoid(self.beta_logit)
        states = delta_states(key_l, value, beta)
        return torch.einsum("bhtdv,bhtd->bhtv", states, query_l)

    @torch.no_grad()
    def acceptance_rate(self, x: torch.Tensor) -> float | None:
        """Hard-threshold acceptance on this block's own ring, or ``None``.

        Expanded has no gate, so it returns ``None`` instead of another
        method's rate. There is no key-id ``hit_rate`` here.
        """
        if self.variant != "recency":
            return None
        if x.ndim != 3:
            raise ValueError(f"expected (batch, time, d_model), got {tuple(x.shape)}")
        batch, time, _ = x.shape
        qkv = self.qkv(x).view(batch, time, 3, self.n_head, self.d_head)
        query, key, _value = (tensor.permute(0, 2, 1, 3) for tensor in qkv.unbind(dim=2))
        scores = torch.einsum("bhtd,bhsd->bhts", _unit(query), _unit(key))
        valid = causal_band(time, self.archive_cap, x.device)
        best = scores.masked_fill(~valid, -1e9).amax(dim=-1)
        tau = torch.tanh(self.tau_logit)
        return float((best >= tau.view(1, -1, 1)).float().mean())

    def gate_fields(self, x: torch.Tensor) -> dict[str, float | str]:
        """Per-variant diagnostic. Expanded's dict has no acceptance field."""
        if self.variant != "recency":
            return {"variant": self.variant}
        rate = self.acceptance_rate(x)
        if rate is None:
            raise RuntimeError("recency acceptance_rate returned None")
        return {"variant": "recency", "accepted_rate": rate}
