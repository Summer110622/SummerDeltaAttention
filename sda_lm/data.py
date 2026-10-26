"""Tasks: multi-query associative recall (MQAR) and byte-level language modelling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def mqar_batch(batch: int, n_pairs: int, n_keys: int = 64, n_values: int = 64,
               rng: np.random.Generator | None = None):
    """Zoology-style MQAR.

    Sequence: k1 v1 k2 v2 ... kN vN | kq1 vq1 kq2 vq2 ... (queries in random
    order, each key queried once). Loss is taken only on the query values.
    Token ids: keys [0, n_keys), values [n_keys, n_keys + n_values).
    Returns inputs (B, T), targets (B, T) with -100 where unsupervised.
    """
    rng = rng or np.random.default_rng()
    T = 4 * n_pairs
    x = np.zeros((batch, T + 1), dtype=np.int64)
    for b in range(batch):
        keys = rng.choice(n_keys, n_pairs, replace=False)
        vals = rng.integers(0, n_values, n_pairs) + n_keys
        order = rng.permutation(n_pairs)
        ctx = np.stack([keys, vals], 1).reshape(-1)
        qry = np.stack([keys[order], vals[order]], 1).reshape(-1)
        x[b, :2 * n_pairs] = ctx
        x[b, 2 * n_pairs:2 * n_pairs + 2 * n_pairs] = qry
    inp, tgt = x[:, :T], x[:, 1:T + 1].copy()
    supervise = np.zeros(T, dtype=bool)
    supervise[2 * n_pairs::2] = True   # position of each query key -> predict value
    tgt[:, ~supervise] = -100
    return torch.from_numpy(inp), torch.from_numpy(tgt)


def load_text_corpus(paths: list[Path], max_bytes: int = 4_000_000) -> np.ndarray:
    buf = bytearray()
    for p in paths:
        try:
            buf += p.read_bytes()
        except OSError:
            continue
        if len(buf) >= max_bytes:
            break
    return np.frombuffer(bytes(buf[:max_bytes]), dtype=np.uint8).astype(np.int64)


def lm_batch(data: np.ndarray, batch: int, T: int, rng: np.random.Generator):
    starts = rng.integers(0, len(data) - T - 1, batch)
    x = np.stack([data[s:s + T + 1] for s in starts])
    return torch.from_numpy(x[:, :-1]), torch.from_numpy(x[:, 1:])
