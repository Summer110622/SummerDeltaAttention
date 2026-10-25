"""Reconstructed hard face (rewrite + distractor) for SDA-NG smoke.

``research/eqmem_hard.py`` and ``research/eqmem_hard_v2/run_v2.py`` are not on
``best-face-results``. This generator uses the constants that *are* documented
(n=2048, U=⌊0.85 n⌋, D=32, V=16, ρ=0.5 → B=1024, Q=128, σ=0 exact keys) and the
qualitative layout "distractors and rewrites sit in the latter half".

It is not a bit-exact replay of the published TEST table. In particular it does
not target rewrite_hit=0.875 or MSE=0.575. Those figures stay in
``research/eqmem_hard_v2/SUMMARY.md``. Same-seed hard-vs-soft gaps in the smoke
log are about this reconstruction only.

Layout at the defaults (deterministic counts, random membership):

* U = 1740 distinct unit keys, R = n − U = 308 rewrite keys, 1432 once-only keys
* first half (1024 writes): early-stable keys, one write each
* second half (1024 writes): 408 distractors once + each rewrite key twice
* the later of the two rewrite writes carries the new value
* queries: 64 rewrite ids + 64 once-only ids, q = k (σ=0), target = latest value

Because every second-half id is more recent than every early id, and
308 + 408 = 716 ≤ B = 1024, Recency-B keeps all rewrite keys, all distractors,
and the 308 most recent early-stable keys.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

N = 2048
D = 32
V = 16
RHO = 0.5
UNIQUE_FRAC = 0.85
Q = 128
EARLY_FRACTION = 0.5


def make_hard_data(seed: int, *, n: int = N, unique_frac: float = UNIQUE_FRAC,
                   rho: float = RHO, n_queries: int = Q,
                   early_fraction: float = EARLY_FRACTION,
                   key_dim: int = D, value_dim: int = V) -> dict:
    """One hard-face stream and an exact-key query set. ``seed`` owns the RNG."""
    if n_queries < 2 or n_queries % 2 != 0:
        raise ValueError("n_queries must be a positive even integer")
    rng = np.random.default_rng(seed)
    n_unique = int(np.floor(unique_frac * n))
    capacity = int(round(rho * n))
    n_rewrite = n - n_unique
    early_slots = int(round(early_fraction * n))
    n_once = n_unique - n_rewrite
    if n_rewrite <= 0 or n_once < early_slots:
        raise ValueError("hard-face counts do not fit this n / unique_frac / early_fraction")
    n_distractor = n_once - early_slots
    if early_slots + n_distractor + 2 * n_rewrite != n:
        raise ValueError("stream length does not match the hard-face partition")
    if n_queries // 2 > n_rewrite or n_queries // 2 > n_once:
        raise ValueError("not enough rewrite or once-only keys to fill the query split")

    perm = rng.permutation(n_unique)
    rewrite_ids = np.asarray(perm[:n_rewrite], dtype=np.int64)
    once_ids = np.asarray(perm[n_rewrite:], dtype=np.int64)
    early_ids = np.asarray(once_ids[:early_slots], dtype=np.int64)
    distractor_ids = np.asarray(once_ids[early_slots:], dtype=np.int64)

    key_table = rng.normal(size=(n_unique, key_dim))
    key_table /= np.linalg.norm(key_table, axis=1, keepdims=True)
    first_value = rng.normal(size=(n_unique, value_dim))
    rewritten = rng.normal(size=(n_rewrite, value_dim))
    new_value_of = {int(key): rewritten[t] for t, key in enumerate(rewrite_ids.tolist())}

    early_order = rng.permutation(early_ids)
    second = np.concatenate([distractor_ids, rewrite_ids, rewrite_ids.copy()])
    rng.shuffle(second)
    key_ids = np.concatenate([early_order, second]).astype(np.int64, copy=False)
    if len(key_ids) != n:
        raise RuntimeError("stream length drifted")

    totals = Counter(int(i) for i in key_ids.tolist())
    seen: Counter[int] = Counter()
    values = np.empty((n, value_dim), dtype=np.float64)
    latest = np.empty((n_unique, value_dim), dtype=np.float64)
    for t, raw in enumerate(key_ids.tolist()):
        key = int(raw)
        seen[key] += 1
        if key in new_value_of and seen[key] == totals[key]:
            value = new_value_of[key]
        else:
            value = first_value[key]
        values[t] = value
        latest[key] = value

    query_rewrite = rng.choice(rewrite_ids, size=n_queries // 2, replace=False)
    query_once = rng.choice(once_ids, size=n_queries // 2, replace=False)
    query_ids = np.concatenate([query_rewrite, query_once]).astype(np.int64, copy=False)
    kind = np.array(["rewrite"] * (n_queries // 2) + ["once"] * (n_queries // 2))
    order = rng.permutation(n_queries)
    query_ids = query_ids[order]
    kind = kind[order]

    return {
        "seed": int(seed),
        "n": int(n),
        "U": int(n_unique),
        "B": int(capacity),
        "D": int(key_dim),
        "V": int(value_dim),
        "n_rewrite_keys": int(n_rewrite),
        "n_early": int(early_slots),
        "n_distractor": int(n_distractor),
        "rewrite_ids": rewrite_ids,
        "early_ids": early_ids,
        "distractor_ids": distractor_ids,
        "once_ids": once_ids,
        "key_ids": key_ids,
        "keys": key_table[key_ids],
        "values": values,
        "query_ids": query_ids,
        "query_keys": key_table[query_ids],
        "targets": latest[query_ids],
        "query_kind": kind,
    }
