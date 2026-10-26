"""Train/evaluate SDA-RC and baselines.

  python -m sda_lm.train --task mqar --mixer sda --steps 1500
  python -m sda_lm.train --task lm --mixer sda --corpus-glob '/usr/lib/python3*/**/*.py'

Writes one JSON line per run to --out.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import lm_batch, load_text_corpus, mqar_batch
from .model import SDARCConfig, SDARCLM


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["mqar", "lm"], default="mqar")
    p.add_argument("--mixer", default="sda")
    p.add_argument("--no-reservoir", action="store_true")
    p.add_argument("--budget", type=int, default=None, help="archive / KV budget")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--d-key", type=int, default=16)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--pairs", type=int, default=16)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--corpus-glob", default="/usr/lib/python3*/**/*.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--out", default="sda_lm/results/runs.jsonl")
    a = p.parse_args(argv)

    torch.manual_seed(a.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rng = np.random.default_rng(a.seed)
    if a.task == "lm":
        files = sorted(Path(f) for f in glob.glob(a.corpus_glob, recursive=True))
        data = load_text_corpus(files)
        split = int(0.9 * len(data))
        train_data, val_data = data[:split], data[split:]
        vocab = 256
    else:
        vocab = 128
    cfg = SDARCConfig(vocab_size=vocab, d_model=a.d_model, n_layers=a.layers,
                      n_heads=a.heads, d_key=a.d_key, d_value=a.d_model // a.heads,
                      mixer=a.mixer, use_reservoir=not a.no_reservoir,
                      archive_budget=a.budget, seed=a.seed)
    model = SDARCLM(cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))

    def batch(r):
        if a.task == "mqar":
            return mqar_batch(a.batch, a.pairs, rng=r)
        return lm_batch(train_data, a.batch, a.seq, r)

    t0 = time.time()
    model.train()
    for step in range(a.steps):
        x, y = batch(rng)
        logits = model(x, archive_seed=int(rng.integers(1 << 31)))
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == a.steps - 1:
            print(f"[{a.mixer}{'' if cfg.use_reservoir else '-noRC'}] step {step} loss {loss.item():.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    erng = np.random.default_rng(10_000 + a.seed)  # held-out eval stream
    tot_loss = tot_n = correct = 0.0
    with torch.no_grad():
        for i in range(a.eval_batches):
            if a.task == "mqar":
                x, y = mqar_batch(a.batch, a.pairs, rng=erng)
            else:
                x, y = lm_batch(val_data, a.batch, a.seq, erng)
            logits = model(x, archive_seed=50_000 + i)
            m = y != -100
            tot_loss += F.cross_entropy(logits[m], y[m], reduction="sum").item()
            correct += (logits[m].argmax(-1) == y[m]).sum().item()
            tot_n += m.sum().item()
    res = {"task": a.task, "mixer": a.mixer, "reservoir": cfg.use_reservoir,
           "budget": a.budget, "pairs": a.pairs if a.task == "mqar" else None,
           "seq": a.seq if a.task == "lm" else 4 * a.pairs, "steps": a.steps,
           "seed": a.seed, **model.param_counts(),
           "eval_loss": tot_loss / tot_n, "eval_acc": correct / tot_n,
           "bits_per_byte": tot_loss / tot_n / math.log(2) if a.task == "lm" else None,
           "train_seconds": round(time.time() - t0, 1)}
    print(json.dumps(res))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    return res


if __name__ == "__main__":
    main()
