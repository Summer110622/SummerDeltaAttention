"""SDA+ : an opt-in, compressed-archive variant of the SDA reference (NumPy only).

This module does not change :mod:`summer_delta_attention.reference`. It keeps the
same no-decay delta-rule fast state and bounded reservoir archive, and adds four
independent, individually switchable pieces:

* **Quantized archive** (``bits``): keys and values are stored as signed integer
  codes with one float16 scale per vector, so a fixed byte budget holds more
  original KV pairs. For ``bits < 8`` the codes are held in an int8 array for
  simplicity; ``bytes_of_state_and_archive`` accounts them as *packed* bits.
* **Clustered read** (``probe``, ``n_clusters``): archived keys are grouped by
  spherical k-means and a query scans only the ``probe`` nearest clusters.
  The index is rebuilt lazily after writes. With ``probe >= n_clusters`` the read
  is an exact scan of the archive.
* **Soft confidence gate** (``gate_temperature``): instead of a hard accept/reject
  at ``threshold``, the archive answer and the fast-state answer are blended with
  ``sigmoid((top_similarity - threshold) / gate_temperature)``.
* **Top-m softmax correction** (``top_m``): the archive answer is a softmax mixture
  of the ``top_m`` most similar archived values instead of the single nearest one.

Honest scope note
-----------------
In the equal-memory synthetic benchmark (``research/plus_equal_memory.py``) the gain
comes from *capacity* (more archived pairs per byte). Setting ``use_state=False`` and
``gated=False`` gives a plain quantized, clustered KV-archive baseline, which scored
about the same as the full variant there. Read the results as "more retained KV per
byte", not as evidence that the delta state or the gate helps.
"""
from __future__ import annotations

import numpy as np


def _qmax(bits: int) -> int:
    return 2 ** (bits - 1) - 1


class SummerDeltaAttentionPlus:
    """Delta-rule fast state plus a quantized, clustered, softly gated KV archive.

    Parameters
    ----------
    key_dim, value_dim: Positive vector dimensions.
    capacity: Maximum number of archived KV pairs (0 is permitted). Use
        :meth:`capacity_for_budget` to derive it from a byte budget.
    bits: Archive code width in ``2..8``, or ``None`` for unquantized storage in
        ``dtype``.
    beta: Delta update step in (0, 1].
    threshold: Similarity at which the soft gate is 0.5.
    gate_temperature: Sigmoid temperature; ``0`` gives a hard gate.
    gated: If ``False`` the archive answer is always used (no fast-state blend).
    use_state: If ``False`` the delta state is not maintained (stays zero).
    top_m: Number of nearest archived entries mixed by softmax.
    probe: Number of clusters scanned per read.
    n_clusters: Cluster count; default ``int(sqrt(archive_size))``.
    softmax_scale: Inverse temperature of the top-m softmax.
    seed: Seed for reservoir sampling and k-means initialisation.
    dtype: Floating type of the fast state and of dequantized values.
    """

    _KMEANS_ITERS = 8
    _MIN_INDEXED = 64  # below this many archived pairs an exact scan is used

    def __init__(self, key_dim: int, value_dim: int, *, capacity: int = 32,
                 bits: int | None = 8, beta: float = 0.05, threshold: float = 0.5,
                 gate_temperature: float = 0.05, gated: bool = True,
                 use_state: bool = True, top_m: int = 8, probe: int = 4,
                 n_clusters: int | None = None, softmax_scale: float = 32.0,
                 seed: int = 0, dtype=np.float64):
        if key_dim <= 0 or value_dim <= 0 or capacity < 0:
            raise ValueError("Dimensions must be positive and capacity nonnegative")
        if bits is not None and not (isinstance(bits, int) and 2 <= bits <= 8):
            raise ValueError("bits must be None or an integer in 2..8")
        if not (0 < beta <= 1):
            raise ValueError("Require 0 < beta <= 1")
        if gate_temperature < 0 or softmax_scale <= 0:
            raise ValueError("Require gate_temperature >= 0 and softmax_scale > 0")
        if top_m < 1 or probe < 1 or (n_clusters is not None and n_clusters < 1):
            raise ValueError("top_m, probe and n_clusters must be positive")
        self.key_dim, self.value_dim, self.capacity = key_dim, value_dim, capacity
        self.bits, self.beta, self.threshold = bits, beta, threshold
        self.gate_temperature, self.gated, self.use_state = gate_temperature, gated, use_state
        self.top_m, self.probe, self.n_clusters = top_m, probe, n_clusters
        self.softmax_scale = softmax_scale
        self.dtype = np.dtype(dtype)
        self.state = np.zeros((key_dim, value_dim), dtype=self.dtype)
        code_dtype = np.int8 if bits is not None else self.dtype
        self._kq = np.zeros((capacity, key_dim), dtype=code_dtype)
        self._vq = np.zeros((capacity, value_dim), dtype=code_dtype)
        self._ks = np.ones(capacity, dtype=np.float16)
        self._vs = np.ones(capacity, dtype=np.float16)
        self._slot_key: list[bytes | None] = [None] * capacity
        self._slot_of: dict[bytes, int] = {}
        self._size = 0
        self._seen = 0
        self._rng = np.random.default_rng(seed)
        self._dirty = True
        self._cent: np.ndarray | None = None
        self._members: list[np.ndarray] = []

    # ------------------------------------------------------------------ utils
    def _unit(self, x, dim: int) -> np.ndarray:
        x = np.asarray(x, dtype=self.dtype)
        if x.shape != (dim,):
            raise ValueError(f"Expected vector shape {(dim,)}, got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("Vector must be finite")
        peak = np.max(np.abs(x))
        if peak == 0:
            raise ValueError("Vector must have nonzero norm")
        x = x / peak
        return x / np.linalg.norm(x)

    def _encode(self, x: np.ndarray):
        if self.bits is None:
            return x.astype(self.dtype, copy=True), np.float16(1.0)
        qmax = _qmax(self.bits)
        scale = np.float16(max(float(np.max(np.abs(x))) / qmax, 6.2e-5))
        codes = np.clip(np.rint(x / float(scale)), -qmax, qmax).astype(np.int8)
        return codes, scale

    def _decode(self, codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
        if self.bits is None:
            return codes.astype(self.dtype)
        return codes.astype(self.dtype) * scales.astype(self.dtype)[..., None]

    # ------------------------------------------------------------------ write
    def write(self, key, value) -> None:
        """Process one token: update the fast state and maybe archive the pair."""
        k = self._unit(key, self.key_dim)
        v = np.asarray(value, dtype=self.dtype)
        if v.shape != (self.value_dim,) or not np.isfinite(v).all():
            raise ValueError("Value must be a finite vector of value_dim")
        if self.use_state:
            self.state += self.beta * np.outer(k, v - k @ self.state)
        self._seen += 1
        if self.capacity == 0:
            return
        kb = k.tobytes()
        slot = self._slot_of.get(kb)  # exact rewrite of an archived key: latest wins
        if slot is None:
            if self._size < self.capacity:
                slot = self._size
                self._size += 1
            else:  # Algorithm R reservoir sampling
                j = int(self._rng.integers(0, self._seen))
                if j >= self.capacity:
                    return
                slot = j
                del self._slot_of[self._slot_key[slot]]
            self._slot_key[slot] = kb
            self._slot_of[kb] = slot
        self._kq[slot], self._ks[slot] = self._encode(k)
        self._vq[slot], self._vs[slot] = self._encode(v)
        self._dirty = True

    # ------------------------------------------------------------------ index
    def _ensure_index(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        n = self._size
        c = self.n_clusters or max(1, int(n ** 0.5))
        if n < self._MIN_INDEXED or c <= 1 or c <= self.probe:
            self._cent, self._members = None, []
            return
        z = self._decode(self._kq[:n], self._ks[:n])
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
        cent = z[self._rng.choice(n, c, replace=False)].copy()
        for _ in range(self._KMEANS_ITERS):
            a = np.argmax(z @ cent.T, axis=1)
            sums = np.zeros_like(cent)
            np.add.at(sums, a, z)
            counts = np.bincount(a, minlength=c)
            norms = np.linalg.norm(sums, axis=1, keepdims=True)
            cent = np.where(counts[:, None] > 0, sums / (norms + 1e-12), cent)
        a = np.argmax(z @ cent.T, axis=1)
        order = np.argsort(a, kind="stable")
        bounds = np.cumsum(np.bincount(a, minlength=c))[:-1]
        self._cent, self._members = cent, np.split(order, bounds)

    # ------------------------------------------------------------------- read
    def read(self, query, *, details: bool = False):
        """o(q) = g * archive_answer + (1 - g) * F^T q, with soft gate g."""
        q = self._unit(query, self.key_dim)
        fast = q @ self.state
        out, info = fast.copy(), {"fast": fast, "similarity": float("nan"),
                                  "gate": 0.0, "candidates": 0,
                                  "archive_size": self._size}
        if self._size:
            self._ensure_index()
            if self._cent is None:
                idx = np.arange(self._size)
            else:
                c = len(self._cent)
                near = np.argpartition(-(self._cent @ q), min(self.probe, c) - 1)[:self.probe]
                idx = np.concatenate([self._members[i] for i in near])
                if len(idx) == 0:
                    idx = np.arange(self._size)
            sims = self._decode(self._kq[idx], self._ks[idx]) @ q
            m = min(self.top_m, len(idx))
            sel = np.argpartition(-sims, m - 1)[:m]
            s = sims[sel]
            w = np.exp(self.softmax_scale * (s - s.max()))
            w /= w.sum()
            chosen = idx[sel]
            archive = w @ self._decode(self._vq[chosen], self._vs[chosen])
            top = float(s.max())
            if not self.gated:
                gate = 1.0
            elif self.gate_temperature > 0:
                gate = float(0.5 * (1.0 + np.tanh((top - self.threshold)
                                                  / (2.0 * self.gate_temperature))))
            else:
                gate = float(top >= self.threshold)
            out = gate * archive + (1.0 - gate) * fast
            info.update(similarity=top, gate=gate, candidates=int(len(idx)))
        return (out, info) if details else out

    # --------------------------------------------------------------- accounting
    @property
    def archive_size(self) -> int:
        return self._size

    @staticmethod
    def entry_bytes(key_dim: int, value_dim: int, bits: int | None,
                    dtype=np.float64) -> float:
        """Payload bytes of one archived pair (codes packed to ``bits`` + 2 fp16 scales)."""
        if bits is None:
            return (key_dim + value_dim) * np.dtype(dtype).itemsize
        return (key_dim + value_dim) * bits / 8 + 4

    @property
    def bytes_of_state_and_archive(self) -> float:
        """Payload only: state + packed archive codes + scales. Excludes the index."""
        state = self.key_dim * self.value_dim * self.dtype.itemsize if self.use_state else 0
        return state + self._size * self.entry_bytes(
            self.key_dim, self.value_dim, self.bits, self.dtype)

    @property
    def bytes_of_index(self) -> int:
        """float32 centroids plus int32 member ids of the current cluster index."""
        self._ensure_index()
        if self._cent is None:
            return 0
        return len(self._cent) * self.key_dim * 4 + self._size * 4

    @classmethod
    def capacity_for_budget(cls, budget_bytes: float, key_dim: int, value_dim: int, *,
                            bits: int | None = 8, use_state: bool = True,
                            dtype=np.float64) -> int:
        """Largest ``capacity`` whose payload + index fits in ``budget_bytes``."""
        state = key_dim * value_dim * np.dtype(dtype).itemsize if use_state else 0
        per = cls.entry_bytes(key_dim, value_dim, bits, dtype)

        def total(cap: int) -> float:
            if cap < cls._MIN_INDEXED:
                return state + cap * per
            c = max(1, int(cap ** 0.5))
            return state + cap * per + c * key_dim * 4 + cap * 4

        lo, hi = 0, int(max(budget_bytes - state, 0) // per) + 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if total(mid) <= budget_bytes else (lo, mid - 1)
        return lo
