"""Tiny synthetic MQAR batch for a causal next-token smoke loop.

This is a data source for ``train_smoke.py``, not a recall benchmark and not
a language-modeling corpus.

Two query mixes share the ``k v`` prefix:

* ``make_mqar_batch`` is the ordinary schedule. The pair index is uniform.
  The query token copies that pair's key. Train and the A3 smoke use this.
* ``make_mqar_exact_key_batch`` is the CALIB eval mix. ``pi_exact`` is 1 and
  ``sigma`` is 0: every query token is a source key, with no noise channel.
  The source is a pair still inside ``source_cap`` when one exists.
"""
from __future__ import annotations

import torch


def _check_shape(batch_size: int, n_pairs: int, n_queries: int, vocab_size: int) -> None:
    if batch_size < 1 or n_pairs < 1 or n_queries < 1:
        raise ValueError("batch_size, n_pairs and n_queries must be positive")
    if vocab_size < 2:
        raise ValueError("vocab_size must be >= 2")


def query_token_positions(n_pairs: int, n_queries: int) -> list[int]:
    """Index of each query token in the full ``k v ... q a ...`` sequence."""
    return [2 * n_pairs + 2 * index for index in range(n_queries)]


def in_ring_pair_indices(n_pairs: int, query_index: int, source_cap: int) -> list[int]:
    """Pair indices whose key position is strictly inside the past ring.

    The ring test matches ``causal_band``: distance ``qpos - key_pos`` is
    positive and strictly less than ``source_cap``. The current position is
    the query, so it is not a source.
    """
    if n_pairs < 1 or query_index < 0 or source_cap < 1:
        raise ValueError("n_pairs, query_index and source_cap are out of range")
    qpos = 2 * n_pairs + 2 * query_index
    return [index for index in range(n_pairs) if 0 < (qpos - 2 * index) < source_cap]


def _nearest_pair_index(n_pairs: int, query_index: int) -> int:
    """Latest pair key that still sits strictly before this query."""
    qpos = 2 * n_pairs + 2 * query_index
    index = min(n_pairs - 1, (qpos - 1) // 2)
    if index < 0 or 2 * index >= qpos:
        raise ValueError("query has no earlier pair key")
    return index


def _pack(keys: torch.Tensor, values: torch.Tensor, which: torch.Tensor
          ) -> tuple[torch.Tensor, torch.Tensor]:
    query = torch.gather(keys, 1, which)
    answer = torch.gather(values, 1, which)
    batch_size, n_pairs = keys.shape
    n_queries = which.shape[1]
    pairs = torch.stack((keys, values), dim=-1).reshape(batch_size, 2 * n_pairs)
    queries = torch.stack((query, answer), dim=-1).reshape(batch_size, 2 * n_queries)
    sequence = torch.cat((pairs, queries), dim=1)
    inputs = sequence[:, :-1].contiguous()
    targets = sequence[:, 1:].contiguous()
    positions = query_token_positions(n_pairs, n_queries)
    if not torch.equal(inputs[:, positions], query):
        raise RuntimeError("query token was dropped from the causal inputs")
    if not torch.equal(targets[:, positions], answer):
        raise RuntimeError("answer token is not the target at the query position")
    return inputs, targets


def make_mqar_batch(batch_size: int = 4, n_pairs: int = 8, n_queries: int = 2,
                    vocab_size: int = 1024, *,
                    generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Build inputs and next-token targets.

    Each row has the layout ``k v`` repeated ``n_pairs`` times, then ``q a``
    repeated ``n_queries`` times. ``q`` copies one of the keys in that row and
    ``a`` is the value written next to it. The pair index is uniform on
    ``[0, n_pairs)``. The returned tensors drop the last input token and the
    first target token so they line up for causal cross-entropy. Token ids
    are uniform integers in ``[0, vocab_size)``.
    """
    _check_shape(batch_size, n_pairs, n_queries, vocab_size)
    keys = torch.randint(0, vocab_size, (batch_size, n_pairs), generator=generator)
    values = torch.randint(0, vocab_size, (batch_size, n_pairs), generator=generator)
    which = torch.randint(0, n_pairs, (batch_size, n_queries), generator=generator)
    return _pack(keys, values, which)


def make_mqar_exact_key_batch(batch_size: int = 4, n_pairs: int = 8, n_queries: int = 2,
                              vocab_size: int = 1024, *,
                              source_cap: int,
                              pi_exact: float = 1.0,
                              sigma: float = 0.0,
                              generator: torch.Generator | None = None,
                              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CALIB eval mix: every query is an exact copy of a source key.

    ``pi_exact`` must be 1 and ``sigma`` must be 0. This function does not
    implement a noisy query channel. The pair prefix uses the same two
    uniform draws as ``make_mqar_batch``. The pair index is then chosen from
    keys that are still inside ``source_cap`` at that query. If that set is
    empty, the query copies the latest earlier key and the caller can see
    that the distance is not inside the cap.

    Returns inputs, targets, and ``which`` of shape ``(batch, n_queries)``.
    """
    if pi_exact != 1.0:
        raise ValueError("exact-key eval mix locks pi_exact at 1.0; a partial mix is not implemented")
    if sigma != 0.0:
        raise ValueError("exact-key eval mix locks sigma at 0; query tokens are not noised")
    if source_cap < 1:
        raise ValueError("source_cap must be >= 1")
    _check_shape(batch_size, n_pairs, n_queries, vocab_size)
    keys = torch.randint(0, vocab_size, (batch_size, n_pairs), generator=generator)
    values = torch.randint(0, vocab_size, (batch_size, n_pairs), generator=generator)
    which = torch.empty(batch_size, n_queries, dtype=torch.long)
    for query_index in range(n_queries):
        candidates = in_ring_pair_indices(n_pairs, query_index, source_cap)
        if candidates:
            pool = torch.tensor(candidates, dtype=torch.long)
            draw = torch.randint(0, pool.numel(), (batch_size,), generator=generator)
            which[:, query_index] = pool[draw]
        else:
            nearest = _nearest_pair_index(n_pairs, query_index)
            qpos = 2 * n_pairs + 2 * query_index
            if (qpos - 2 * nearest) < source_cap:
                raise RuntimeError("nearest key was inside the cap but was not a candidate")
            which[:, query_index] = nearest
    query = torch.gather(keys, 1, which)
    inputs, targets = _pack(keys, values, which)
    positions = query_token_positions(n_pairs, n_queries)
    if not torch.equal(inputs[:, positions], query):
        raise RuntimeError("eval inputs do not hold the exact source keys")
    return inputs, targets, which
