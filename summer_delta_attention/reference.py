"""Educational NumPy reference for SDA v5-Gated (not an optimized kernel).

A no-decay delta-rule state is supplemented by a *bounded*, reservoir-sampled
key-value archive. Queries scan that archive exactly and correct the fast-state
prediction only when the top cosine similarity clears a confidence threshold.

This is an inference-time associative-memory prototype, not a trained model.
The archive scan is O(B*d_k) per query; B can grow with context length.

Behavioural notes (see the README errata for the review that motivated them):

* Writing a key that is (numerically) identical to an archived key *overwrites*
  that slot's value instead of adding a duplicate. Without this, an exact-key
  query returns the oldest value for the key, i.e. a stale one, and the
  correction term cancels the delta-rule state that had already been updated.
* For an archived key the correction makes the output equal the archived value
  exactly, so the fast state only matters for rejected or missing keys.
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
    seed: Seed for reservoir sampling. The stream's *fast state* is always
          updated; archived pairs are a fixed-size sample of the stream
          (uniform over writes whose key is not already archived).
    dtype: NumPy floating type (float64 used for numerical reproducibility).
    overwrite_tol: A written key whose cosine with an archived key is at least
          ``1 - overwrite_tol`` replaces that slot's value (latest write wins).
    """

    def __init__(self, key_dim: int, value_dim: int, *, capacity: int = 32,
                 beta: float = 0.05, threshold: float = 0.75,
                 seed: int = 0, dtype=np.float64, overwrite_tol: float = 1e-9):
        if key_dim <= 0 or value_dim <= 0 or capacity < 0:
            raise ValueError("Dimensions must be positive and capacity nonnegative")
        if not (0 < beta <= 1) or not (-1 <= threshold <= 1):
            raise ValueError("Require 0 < beta <= 1 and -1 <= threshold <= 1")
        if not (0 <= overwrite_tol < 1):
            raise ValueError("Require 0 <= overwrite_tol < 1")
        self.key_dim, self.value_dim = key_dim, value_dim
        self.capacity, self.beta, self.threshold = capacity, beta, threshold
        self.overwrite_tol = overwrite_tol
        self.dtype = np.dtype(dtype)
        self.state = np.zeros((key_dim, value_dim), dtype=self.dtype)
        # Preallocated archive: reads slice views instead of re-stacking lists.
        self._keys = np.zeros((capacity, key_dim), dtype=self.dtype)
        self._values = np.zeros((capacity, value_dim), dtype=self.dtype)
        self._size = 0
        self._seen = 0
        self._rng = np.random.default_rng(seed)

    def _unit(self, x: np.ndarray, dim: int) -> np.ndarray:
        x = np.asarray(x, dtype=self.dtype)
        if x.shape != (dim,):
            raise ValueError(f"Expected vector shape {(dim,)}, got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("Vector must be finite")
        peak = np.max(np.abs(x))
        if peak == 0:
            raise ValueError("Vector must have nonzero norm")
        # Rescale before taking the norm so huge finite entries cannot overflow.
        x = x / peak
        return x / np.linalg.norm(x)

    def _matching_slot(self, k: np.ndarray):
        if self._size == 0:
            return None
        scores = self._keys[:self._size] @ k
        slot = int(np.argmax(scores))
        return slot if scores[slot] >= 1.0 - self.overwrite_tol else None

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
        slot = self._matching_slot(k)
        if slot is None:
            if self._size < self.capacity:
                slot = self._size
                self._size += 1
            else:
                # Reservoir sampling (Algorithm R): each of the `seen` tokens is
                # equally likely to occupy one of the `capacity` slots.
                j = int(self._rng.integers(0, self._seen))
                if j >= self.capacity:
                    return
                slot = j
        # Overwrite of an archived key always lands here, so the archive never
        # serves a stale value for a key that has been rewritten.
        self._keys[slot] = k
        self._values[slot] = v

    def read(self, query: np.ndarray, *, details: bool = False):
        """Read from the fast state and correct via a confident nearest KV.

        o(q) = F^T q + 1[cos(q,k*) >= tau] (v* - F^T k*).
        """
        q = self._unit(query, self.key_dim)
        fast = q @ self.state
        score, accepted = float("nan"), False
        out = fast.copy()
        if self._size:
            keys = self._keys[:self._size]
            scores = keys @ q
            index = int(np.argmax(scores))
            score = float(scores[index])
            accepted = score >= self.threshold
            if accepted:
                out += self._values[index] - keys[index] @ self.state
        if details:
            return out, {"fast": fast, "similarity": score,
                         "accepted": accepted, "archive_size": self._size}
        return out

    def read_batch(self, queries: np.ndarray, *, batch_size: int = 256,
                   archive_block_size: int = 4096, details: bool = False):
        """Read independent queries against the current state using matrix products.

        ``queries`` has shape ``(n_queries, key_dim)``; the result has shape
        ``(n_queries, value_dim)``. With ``details=True``, diagnostics contain
        arrays ``fast``, ``similarity`` and ``accepted``, plus scalar
        ``archive_size``. An empty query matrix is supported.

        Both query and archive axes are tiled: the score workspace is at most
        ``batch_size * archive_block_size`` scalars. Every archive entry is
        searched; this is not approximate retrieval. Numerically ambiguous
        gates/nearest-key ties fall back to :meth:`read`. Other results can
        differ by floating-point roundoff from repeated scalar reads.

        Writes must not be interleaved with this call. For online write/read
        workloads, continue using :meth:`read` after each write.
        """
        for name, size in (("batch_size", batch_size),
                           ("archive_block_size", archive_block_size)):
            if (isinstance(size, (bool, np.bool_)) or
                    not isinstance(size, (int, np.integer)) or size <= 0):
                raise ValueError(f"{name} must be a positive integer")
        queries = np.asarray(queries, dtype=self.dtype)
        if queries.ndim != 2 or queries.shape[1] != self.key_dim:
            raise ValueError(f"Expected query matrix shape (n, {self.key_dim})")
        n = len(queries)
        out = np.empty((n, self.value_dim), dtype=self.dtype)
        if details:
            info = {"fast": np.empty_like(out),
                    "similarity": np.full(n, np.nan, dtype=self.dtype),
                    "accepted": np.zeros(n, dtype=bool),
                    "archive_size": self._size}
        # A conservative roundoff band for unit-vector dot products. Decisions
        # in this band use the original scalar path, including its normalization.
        tolerance = 8 * self.key_dim * np.finfo(self.dtype).eps
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            x = queries[start:stop]
            if not np.isfinite(x).all():
                raise ValueError("Vector must be finite")
            peak = np.max(np.abs(x), axis=1, keepdims=True)
            if (peak == 0).any():
                raise ValueError("Vector must have nonzero norm")
            q = x / peak
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            fast = q @ self.state
            result = out[start:stop]
            result[:] = fast
            rows = np.arange(stop - start)
            best = np.full(len(rows), -np.inf, dtype=self.dtype)
            second = np.full_like(best, -np.inf)
            indices = np.zeros(len(rows), dtype=np.intp)
            for offset in range(0, self._size, archive_block_size):
                keys = self._keys[offset:min(offset + archive_block_size, self._size)]
                scores = q @ keys.T
                local_index = np.argmax(scores, axis=1)
                local_best = scores[rows, local_index].copy()
                scores[rows, local_index] = -np.inf
                local_second = np.max(scores, axis=1)
                second = np.maximum(np.maximum(second, local_second),
                                    np.minimum(best, local_best))
                better = local_best > best  # first archive slot wins exact ties
                indices[better] = offset + local_index[better]
                best = np.maximum(best, local_best)
                del scores  # release this tile before allocating the next one
            if self._size:
                accepted = best >= self.threshold
                selected = indices[accepted]
                result[accepted] += (self._values[selected] -
                                     self._keys[selected] @ self.state)
                ambiguous = ((best - second <= tolerance) |
                             (np.abs(best - self.threshold) <= tolerance))
                for row in np.flatnonzero(ambiguous):
                    result[row], scalar = self.read(x[row], details=True)
                    fast[row] = scalar["fast"]
                    best[row] = scalar["similarity"]
                    accepted[row] = scalar["accepted"]
                if details:
                    info["similarity"][start:stop] = best
                    info["accepted"][start:stop] = accepted
            if details:
                info["fast"][start:stop] = fast
        return (out, info) if details else out

    @property
    def archive_size(self) -> int:
        return self._size

    @property
    def bytes_of_state_and_archive(self) -> int:
        """Array payload only, excluding Python/index/allocator overhead."""
        scalars = self.key_dim * self.value_dim
        scalars += self.archive_size * (self.key_dim + self.value_dim)
        return scalars * self.dtype.itemsize


def kda_proxy_state(keys: np.ndarray, values: np.ndarray, *, beta: float = 0.05,
                    decay_rate: float = 0.0005, gate_eta: float = 0.0) -> np.ndarray:
    """Untrained channel-decay delta-rule recurrence, NOT published Kimi Linear.

    Input keys must be L2-normalized (checked). With the default ``gate_eta=0``
    the gate is a fixed per-channel decay profile, i.e. time-invariant; it is a
    limited synthetic baseline only.
    """
    keys, values = np.asarray(keys), np.asarray(values)
    if keys.ndim != 2 or values.ndim != 2 or len(keys) != len(values):
        raise ValueError("Expected aligned rank-two keys and values")
    if not np.allclose(np.linalg.norm(keys, axis=1), 1.0, atol=1e-6):
        raise ValueError("Keys must be L2-normalized")
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
