"""SDA-NG NumPy reference (synthetic recall only).

Skeleton kept from the hard-face winner (SDA-Recency):
  * no-decay delta-rule state F
  * Recency-B archive: LRU over distinct key ids, latest value wins

The hard accept 1[cos >= tau] is replaced by the frozen soft residual weight
in DESIGN.md (案1). The margin ramp (案2) is intentionally not implemented.

``research/eqmem_hard.py`` is not on ``best-face-results``. ``recency_positions``
is the documented Recency-B index rule (latest write of each retained id),
not Algorithm R.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

# Frozen in DESIGN.md. Not re-fit on TEST seeds.
BETA = 0.05
TAU_FROZEN = 0.75
TEMPERATURE = 0.10
T_EPS = 1e-6
QUANTILE_P = 0.30


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic σ(x) = 1 / (1 + exp(−x)), stable for large |x|."""
    z = np.asarray(x, dtype=np.float64)
    scalar = z.ndim == 0
    z = np.atleast_1d(z)
    out = np.empty(z.shape, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    negative = ~positive
    exp_z = np.exp(z[negative])
    out[negative] = exp_z / (1.0 + exp_z)
    if scalar:
        return out[0]
    return out


def _temperature(temperature: float, eps: float) -> float:
    """T = max(temperature, eps). Enforces T > 0 via a floor at eps."""
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    T = float(temperature)
    if not np.isfinite(T):
        raise ValueError("temperature must be finite")
    return float(eps) if T < float(eps) else T


def soft_gate(cos, tau: float = TAU_FROZEN, temperature: float = TEMPERATURE,
              eps: float = T_EPS) -> np.ndarray:
    """g = σ((cos − τ) / T) with T = max(temperature, eps).

    g is continuous and lies in (0, 1) for finite cos.
    """
    T = _temperature(temperature, eps)
    shifted = (np.asarray(cos, dtype=np.float64) - float(tau)) / T
    return sigmoid(shifted)


def hard_gate(cos, tau: float = TAU_FROZEN) -> np.ndarray:
    """SDA-Recency baseline: g = 1[cos ≥ τ]."""
    return (np.asarray(cos, dtype=np.float64) >= float(tau)).astype(np.float64)


def calib_tau(cos, p: float = QUANTILE_P) -> float:
    """Optional CALIB-only threshold: τ ← Quantile_p(cos).

    NumPy's linear quantile (the library default). For a hard gate this
    targets an accept rate near 1 − p when the cosine sample is continuous.
    It is an inference constant fit on CALIB cosines, not a TEST search.
    """
    sample = np.asarray(cos, dtype=np.float64).ravel()
    if sample.size == 0 or not np.isfinite(sample).all():
        raise ValueError("cosine sample must be non-empty and finite")
    if not (0.0 <= float(p) <= 1.0):
        raise ValueError("quantile level p must lie in [0, 1]")
    return float(np.quantile(sample, float(p)))


def gate_from_cosine(cos, mode: str, tau: float, temperature: float = TEMPERATURE,
                     eps: float = T_EPS) -> np.ndarray:
    if mode == "soft":
        return soft_gate(cos, tau=tau, temperature=temperature, eps=eps)
    if mode == "hard":
        return hard_gate(cos, tau=tau)
    raise ValueError("mode must be 'soft' or 'hard'")


def recency_positions(key_ids, capacity: int) -> np.ndarray:
    """Latest-write indices of the ``capacity`` most recent distinct key ids.

    Order is oldest-retained first (LRU order). Keys written more recently
    than the retained set are the only ones kept; a repeated id updates its
    latest position instead of taking a second slot.
    """
    if capacity < 0:
        raise ValueError("capacity must be nonnegative")
    last: dict[int, int] = {}
    for t, raw in enumerate(np.asarray(key_ids).tolist()):
        last[int(raw)] = t
    ordered = sorted(last, key=lambda key: last[key])
    if len(ordered) > capacity:
        ordered = ordered[-capacity:]
    return np.asarray([last[key] for key in ordered], dtype=np.int64)


def residual_output(query, state, key_j, value_j, weight) -> np.ndarray:
    """out = Fᵀq + g · (v_j − Fᵀk_j).

    ``query`` is (Q, d) or (d,), ``state`` is (d, v), ``key_j`` / ``value_j``
    match the query batch, and ``weight`` broadcasts over queries.
    """
    q = np.asarray(query, dtype=np.float64)
    F = np.asarray(state, dtype=np.float64)
    kj = np.asarray(key_j, dtype=np.float64)
    vj = np.asarray(value_j, dtype=np.float64)
    g = np.asarray(weight, dtype=np.float64)
    fast = q @ F
    correction = vj - kj @ F
    if g.ndim == 0:
        return fast + g * correction
    return fast + g[..., None] * correction


class RecencySDA:
    """No-decay delta F plus a Recency-B archive and a residual gate.

    Parameters
    ----------
    key_dim, value_dim:
        Positive vector sizes.
    capacity:
        Recency-B size. 0 keeps F only.
    beta:
        Delta step in (0, 1]. Default is the committed SDA-Recency value.
    tau, temperature, eps:
        Frozen soft-gate parameters. ``read(..., mode='hard')`` ignores
        temperature and uses 1[cos ≥ tau]. Overrides on ``read`` do not
        change the stored defaults.
    """

    def __init__(self, key_dim: int, value_dim: int, *, capacity: int,
                 beta: float = BETA, tau: float = TAU_FROZEN,
                 temperature: float = TEMPERATURE, eps: float = T_EPS):
        if key_dim <= 0 or value_dim <= 0 or capacity < 0:
            raise ValueError("dimensions must be positive and capacity nonnegative")
        if not (0 < float(beta) <= 1):
            raise ValueError("require 0 < beta <= 1")
        if not (-1.0 <= float(tau) <= 1.0):
            raise ValueError("require -1 <= tau <= 1")
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.capacity = int(capacity)
        self.beta = float(beta)
        self.tau = float(tau)
        self.temperature = _temperature(temperature, eps)
        self.eps = float(eps)
        self.state = np.zeros((self.key_dim, self.value_dim), dtype=np.float64)
        # key id -> (unit key, value). Order is oldest -> newest.
        self._items: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def _unit(self, x: np.ndarray) -> np.ndarray:
        vec = np.asarray(x, dtype=np.float64)
        if vec.shape != (self.key_dim,):
            raise ValueError(f"expected key shape {(self.key_dim,)}, got {vec.shape}")
        if not np.isfinite(vec).all():
            raise ValueError("key must be finite")
        peak = float(np.max(np.abs(vec)))
        if peak == 0.0:
            raise ValueError("key must have nonzero norm")
        vec = vec / peak
        return vec / np.linalg.norm(vec)

    def _unit_batch(self, x: np.ndarray) -> np.ndarray:
        mat = np.asarray(x, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[1] != self.key_dim:
            raise ValueError(f"expected query shape (Q, {self.key_dim})")
        if not np.isfinite(mat).all():
            raise ValueError("queries must be finite")
        peak = np.max(np.abs(mat), axis=1, keepdims=True)
        if np.any(peak == 0.0):
            raise ValueError("queries must have nonzero norm")
        mat = mat / peak
        norm = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / norm

    def write(self, key_id: int, key: np.ndarray, value: np.ndarray) -> None:
        """One no-decay delta step, then Recency-B insert/refresh."""
        k = self._unit(key)
        v = np.asarray(value, dtype=np.float64).copy()
        if v.shape != (self.value_dim,) or not np.isfinite(v).all():
            raise ValueError("value must be a finite vector of value_dim")
        error = v - k @ self.state
        self.state += self.beta * np.outer(k, error)
        if self.capacity == 0:
            return
        key_id = int(key_id)
        if key_id in self._items:
            self._items.move_to_end(key_id)
            self._items[key_id] = (k, v)
            return
        self._items[key_id] = (k, v)
        if len(self._items) > self.capacity:
            self._items.popitem(last=False)

    def archived_ids(self) -> list[int]:
        """Retained ids, oldest first."""
        return [int(key) for key in self._items]

    @property
    def archive_size(self) -> int:
        return len(self._items)

    def _matrices(self):
        if not self._items:
            empty_k = np.zeros((0, self.key_dim), dtype=np.float64)
            empty_v = np.zeros((0, self.value_dim), dtype=np.float64)
            empty_i = np.zeros((0,), dtype=np.int64)
            return empty_k, empty_v, empty_i
        ids, keys, values = [], [], []
        for key_id, (key, value) in self._items.items():
            ids.append(key_id)
            keys.append(key)
            values.append(value)
        return (np.stack(keys), np.stack(values),
                np.asarray(ids, dtype=np.int64))

    def read(self, query: np.ndarray, *, mode: str = "soft", tau: float | None = None,
             temperature: float | None = None, details: bool = False):
        """Batch or single-vector read.

        cos_j = max_i ⟨q, k_i⟩ over the archive, then
        out = Fᵀq + g (v_j − Fᵀk_j) with g from ``mode``.
        An empty archive returns Fᵀq and g = 0.
        """
        single = np.asarray(query).ndim == 1
        q = self._unit_batch(query[None, :] if single else query)
        tau = self.tau if tau is None else float(tau)
        temperature = self.temperature if temperature is None else float(temperature)
        fast = q @ self.state
        keys, values, ids = self._matrices()
        if len(ids) == 0:
            cos = np.full(len(q), np.nan, dtype=np.float64)
            weight = np.zeros(len(q), dtype=np.float64)
            neighbor = np.full(len(q), -1, dtype=np.int64)
            out = fast.copy()
        else:
            # cos_j = max_i ⟨q, k_i⟩. Archive keys are stored as rows.
            scores = q @ keys.T
            neighbor_row = np.argmax(scores, axis=1)
            rows = np.arange(len(q))
            cos = scores[rows, neighbor_row].astype(np.float64, copy=False)
            weight = gate_from_cosine(cos, mode, tau, temperature, self.eps)
            out = residual_output(q, self.state, keys[neighbor_row], values[neighbor_row], weight)
            neighbor = ids[neighbor_row]
        if single:
            out = out[0]
        if not details:
            return out
        info = {
            "cosine": cos[0] if single else cos,
            "weight": weight[0] if single else weight,
            "neighbor_id": int(neighbor[0]) if single else neighbor,
            "fast": fast[0] if single else fast,
            "archive_size": self.archive_size,
            "tau": tau,
            "mode": mode,
        }
        return out, info
