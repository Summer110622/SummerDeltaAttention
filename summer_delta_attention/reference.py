"""Educational NumPy reference for SDA v5-Gated (not an optimized kernel).

A no-decay delta-rule state is supplemented by a *bounded*, reservoir-sampled
key-value archive. Queries scan that archive exactly and correct the fast-state
prediction only when the top cosine similarity clears a confidence threshold.

This is an inference-time associative-memory prototype, not a trained model.
The archive scan is O(B*d_k) per query; B can grow with context length.
"""
from __future__ import annotations

import numpy as np


class SummerDeltaAttention:
    """No-decay delta-rule fast weights plus gated indexed KV correction.

    Parameters
    ----------
    key_dim, value_dim: Positive vector dimensions.
    capacity: Maximum number of archived original KV pairs (0 is permitted).
    beta: Delta update step in (0, 1].
    threshold: Cosine-similarity gate threshold in [-1, 1].
    seed: Seed for reservoir sampling. The stream's *fast state* is always updated;
          archived pairs are an unbiased fixed-size sample of the stream.
    dtype: NumPy floating type (float64 used for numerical reproducibility).
    """

    def __init__(self, key_dim: int, value_dim: int, *, capacity: int = 32,
                 beta: float = 0.05, threshold: float = 0.75,
                 seed: int = 0, dtype=np.float64):
        if key_dim <= 0 or value_dim <= 0 or capacity < 0:
            raise ValueError("Dimensions must be positive and capacity nonnegative")
        if not (0 < beta <= 1) or not (-1 <= threshold <= 1):
            raise ValueError("Require 0 < beta <= 1 and -1 <= threshold <= 1")
        self.key_dim, self.value_dim = key_dim, value_dim
        self.capacity, self.beta, self.threshold = capacity, beta, threshold
        self.dtype = np.dtype(dtype)
        self.state = np.zeros((key_dim, value_dim), dtype=self.dtype)
        self._keys: list[np.ndarray] = []
        self._values: list[np.ndarray] = []
        self._seen = 0
        self._rng = np.random.default_rng(seed)

    def _unit(self, x: np.ndarray, dim: int) -> np.ndarray:
        x = np.asarray(x, dtype=self.dtype)
        if x.shape != (dim,):
            raise ValueError(f"Expected vector shape {(dim,)}, got {x.shape}")
        norm = np.linalg.norm(x)
        if not np.isfinite(norm) or norm <= 0 or not np.isfinite(x).all():
            raise ValueError("Vector must be finite and have nonzero norm")
        return x / norm

    def write(self, key: np.ndarray, value: np.ndarray) -> None:
        """Process one token and optionally archive its *original* KV pair."""
        k = self._unit(key, self.key_dim)
        v = np.asarray(value, dtype=self.dtype)
        if v.shape != (self.value_dim,) or not np.isfinite(v).all():
            raise ValueError("Value must be a finite vector of value_dim")
        error = v - k @ self.state
        self.state += self.beta * np.outer(k, error)
        self._seen += 1
        if self.capacity == 0:
            return
        if len(self._keys) < self.capacity:
            self._keys.append(k.copy())
            self._values.append(v.copy())
        else:
            # Reservoir sampling: each of the `seen` tokens is equally likely
            # to occupy one of the `capacity` slots. No learned selector.
            j = int(self._rng.integers(0, self._seen))
            if j < self.capacity:
                self._keys[j], self._values[j] = k.copy(), v.copy()

    def read(self, query: np.ndarray, *, details: bool = False):
        """Read from the fast state and correct via a confident nearest KV.

        The returned prediction uses exactly the SDA-5 paper formula:
        o(q) = F^T q + 1[cos(q,k*) >= tau] (v* - F^T k*).
        """
        q = self._unit(query, self.key_dim)
        fast = q @ self.state
        score, accepted = float("nan"), False
        out = fast.copy()
        if self._keys:
            keys = np.stack(self._keys)
            scores = keys @ q
            index = int(np.argmax(scores))
            score = float(scores[index])
            accepted = score >= self.threshold
            if accepted:
                out += self._values[index] - keys[index] @ self.state
        if details:
            return out, {"fast": fast, "similarity": score,
                         "accepted": accepted, "archive_size": len(self._keys)}
        return out

    @property
    def archive_size(self) -> int:
        return len(self._keys)

    @property
    def bytes_of_state_and_archive(self) -> int:
        """Array payload only, excluding Python/index/allocator overhead."""
        scalars = self.key_dim * self.value_dim
        scalars += self.archive_size * (self.key_dim + self.value_dim)
        return scalars * self.dtype.itemsize


def kda_proxy_state(keys: np.ndarray, values: np.ndarray, *, beta: float = 0.05,
                    decay_rate: float = 0.0005, gate_eta: float = 0.0) -> np.ndarray:
    """Untrained channel-gated KDA-style recurrence, NOT published Kimi Linear.

    Input keys are expected to be L2-normalized. The gate is deterministic,
    not trained. It is a limited synthetic baseline only.
    """
    keys, values = np.asarray(keys), np.asarray(values)
    if keys.ndim != 2 or values.ndim != 2 or len(keys) != len(values):
        raise ValueError("Expected aligned rank-two keys and values")
    d = keys.shape[1]
    state = np.zeros((d, values.shape[1]), dtype=np.float64)
    channel_profile = np.linspace(0.5, 1.5, d)
    for key, value in zip(keys, values):
        k = np.asarray(key, dtype=np.float64)
        a = np.exp(-decay_rate * channel_profile *
                   (1 + gate_eta * np.sqrt(d) * np.abs(k)))
        state *= a[:, None]
        state += beta * np.outer(k, value - k @ state)
    return state


def full_kv_attention(query: np.ndarray, keys: np.ndarray,
                      values: np.ndarray, *, scale: float = 32.0) -> np.ndarray:
    """Sharp softmax KV reference; not an equal-memory trained transformer."""
    q = np.asarray(query, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    scores = scale * (keys @ q)
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    return weights @ values
