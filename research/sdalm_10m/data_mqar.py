"""Tiny synthetic MQAR batch for a causal next-token smoke loop.

This is a data source for ``train_smoke.py``, not a recall benchmark and not
a language-modeling corpus.
"""
from __future__ import annotations

import torch


def make_mqar_batch(batch_size: int = 4, n_pairs: int = 8, n_queries: int = 2,
                    vocab_size: int = 1024, *,
                    generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Build inputs and next-token targets.

    Each row has the layout ``k v`` repeated ``n_pairs`` times, then ``q a``
    repeated ``n_queries`` times. ``q`` copies one of the keys in that row and
    ``a`` is the value written next to it. The returned tensors drop the last
    input token and the first target token so they line up for causal
    cross-entropy. Token ids are uniform integers in ``[0, vocab_size)``.
    """
    if batch_size < 1 or n_pairs < 1 or n_queries < 1:
        raise ValueError("batch_size, n_pairs and n_queries must be positive")
    if vocab_size < 2:
        raise ValueError("vocab_size must be >= 2")
    keys = torch.randint(0, vocab_size, (batch_size, n_pairs), generator=generator)
    values = torch.randint(0, vocab_size, (batch_size, n_pairs), generator=generator)
    which = torch.randint(0, n_pairs, (batch_size, n_queries), generator=generator)
    query = torch.gather(keys, 1, which)
    answer = torch.gather(values, 1, which)
    pairs = torch.stack((keys, values), dim=-1).reshape(batch_size, 2 * n_pairs)
    queries = torch.stack((query, answer), dim=-1).reshape(batch_size, 2 * n_queries)
    sequence = torch.cat((pairs, queries), dim=1)
    return sequence[:, :-1].contiguous(), sequence[:, 1:].contiguous()
