"""Equal-memory synthetic recall benchmark for SDA vs SDA+ (NumPy only).

Run:  OPENBLAS_NUM_THREADS=1 python research/plus_equal_memory.py [--quick]

Setup (synthetic, NOT a language-model benchmark)
-------------------------------------------------
* N unit keys drawn from 64 topic directions (key = unit(topic + unit(noise))),
  random Gaussian values, d_key = d_value = 64.
* Query = unit(key_t + sigma * unit(noise)) for a random target position t.
* Metric: top-1 retrieval = argmax_j cos(output, value_j) == t.
* Every method gets the same byte budget; archive capacity is derived from it
  (float32 for the original SDA, packed int8 / int4 + fp16 scales + the cluster
  index for SDA+). Bytes reported are the measured payload (+index for SDA+).

Methods: original SDA (reservoir archive, hard gate, no quantization), SDA+ with
int8 and int4, and two ablations of SDA+ at int4: no delta state, and no gate
(``use_state=False, gated=False`` = a plain quantized clustered KV archive).
"""
import argparse
import time

import numpy as np

from summer_delta_attention import SummerDeltaAttention, SummerDeltaAttentionPlus as Plus

DK = DV = 64
TOPICS = 64


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def make_task(n, n_queries, sigma, seed, strength=1.0):
    rng = np.random.default_rng(seed)
    topics = unit(rng.normal(size=(TOPICS, DK)))
    keys = unit(strength * topics[rng.integers(0, TOPICS, n)] + unit(rng.normal(size=(n, DK))))
    values = rng.normal(size=(n, DV))
    target = rng.integers(0, n, n_queries)
    queries = unit(keys[target] + sigma * unit(rng.normal(size=(n_queries, DK))))
    return keys, values, queries, target


def top1(outputs, values, target):
    sims = unit(outputs) @ unit(values).T
    return float((sims.argmax(1) == target).mean())


def run_original(keys, values, queries, budget, seed):
    state = DK * DV * 4
    cap = max(int((budget - state) // ((DK + DV) * 4)), 0)
    m = SummerDeltaAttention(DK, DV, capacity=cap, beta=0.05, threshold=0.75,
                             seed=seed, dtype=np.float32)
    for k, v in zip(keys, values):
        m.write(k, v)
    out = np.stack([m.read(q) for q in queries])
    return out, m.bytes_of_state_and_archive, m.archive_size


def run_plus(keys, values, queries, budget, seed, *, bits, use_state=True, gated=True):
    cap = Plus.capacity_for_budget(budget, DK, DV, bits=bits, use_state=use_state,
                                   dtype=np.float32)
    m = Plus(DK, DV, capacity=cap, bits=bits, use_state=use_state, gated=gated,
             seed=seed, dtype=np.float32)
    for k, v in zip(keys, values):
        m.write(k, v)
    out = np.stack([m.read(q) for q in queries])
    return out, m.bytes_of_state_and_archive + m.bytes_of_index, m.archive_size


METHODS = {
    "SDA original (float32)": lambda *a, **k: run_original(*a, **k),
    "SDA+ int8": lambda *a, **k: run_plus(*a, bits=8, **k),
    "SDA+ int4": lambda *a, **k: run_plus(*a, bits=4, **k),
    "SDA+ int4, no state": lambda *a, **k: run_plus(*a, bits=4, use_state=False, **k),
    "SDA+ int4, no state, no gate": lambda *a, **k: run_plus(
        *a, bits=4, use_state=False, gated=False, **k),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small smoke run")
    args = ap.parse_args()
    ns = [2048] if args.quick else [4096, 16384]
    budgets_kib = [64] if args.quick else [272, 2064]
    sigmas = [0.3] if args.quick else [0.3, 1.0]
    seeds = [0] if args.quick else [0, 1]
    n_queries = 64 if args.quick else 128

    print(f"{'N':>6} {'sigma':>5} {'KiB':>5}  {'method':<30} {'top1':>6} {'used KiB':>9} {'archived':>9}")
    for n in ns:
        for sigma in sigmas:
            for kib in budgets_kib:
                acc = {name: [] for name in METHODS}
                info = {}
                for seed in seeds:
                    keys, values, queries, target = make_task(n, n_queries, sigma, seed)
                    for name, fn in METHODS.items():
                        out, used, archived = fn(keys, values, queries, kib * 1024, seed=seed)
                        acc[name].append(top1(out, values, target))
                        info[name] = (used / 1024, archived)
                        assert used <= kib * 1024 + 1, (name, used)
                for name in METHODS:
                    print(f"{n:>6} {sigma:>5} {kib:>5}  {name:<30} "
                          f"{np.mean(acc[name]):>6.3f} {info[name][0]:>9.1f} {info[name][1]:>9d}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[done in {time.time() - t0:.0f}s]")
