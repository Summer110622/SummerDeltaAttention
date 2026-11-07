# Opt-in confidence pruning for SDA reads

`SummerDeltaAttention.read_batch(..., prune_rejected=True)` avoids computing a
runner-up score for tiles whose scores are all certainly below the confidence
gate. Every archive entry is still searched. Accepted matches and gate-boundary
decisions retain the original scalar numerical guards. It also avoids an empty
correction matrix product when no queries are accepted.

The option defaults to **False** because the additional gate checks cost time
when queries hit the archive. This is a workload-specific optimization, not a
universal speedup or a change to the attention recurrence.

## Paired CPU benchmark

Confirmation run `9c19178c-b2db-43f5-a44a-59cbed10d836` used experimental snapshot
`462f3a1df2dd8ae9e1d5dbd3d77e32ad8d953723`. The implementation, batch tests, and
benchmark script in this PR were checked byte-for-byte against that snapshot.

- AMD Ryzen 5 3600 CPU; Python 3.12.12; NumPy 2.5.4; float32.
- One thread each for OpenBLAS, OpenMP, and MKL.
- Key/value dimensions 32/16, threshold 0.75, beta 0.05.
- Archive capacities 1,024/8,192; 4/128 queries; exact-key query fractions
  0/50/100%; seeds 0–4; seven timed repeats.
- Both variants read the **same populated memory**. Setup and writes are
  excluded. Timing order alternates and multiple calls are amortized in each
  repeat. All repeat timings and effective configuration are printed by the
  script and written to `outputs/sda_pruning.json`.

Times below are the medians across seeds of each seed's median repeat time.
Speedup is the median of per-seed baseline/pruned ratios, so it need not equal
the ratio of the two displayed times. The interval is the observed seed range,
not a confidence interval.

| Capacity | Queries | Exact-key fraction | Baseline ms | Pruned ms | Paired speedup | Seed range |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 4 | 0% | 0.07969 | 0.06613 | 1.204× | 1.193–1.210× |
| 1024 | 4 | 50% | 0.08198 | 0.08684 | 0.946× | 0.943–0.957× |
| 1024 | 4 | 100% | 0.07907 | 0.08472 | 0.943× | 0.913–0.946× |
| 1024 | 128 | 0% | 0.20843 | 0.17899 | 1.144× | 1.142–1.164× |
| 1024 | 128 | 50% | 0.22127 | 0.22771 | 0.979× | 0.970–1.009× |
| 1024 | 128 | 100% | 0.22385 | 0.22627 | 0.979× | 0.965–0.994× |
| 8192 | 4 | 0% | 0.16525 | 0.14429 | 1.138× | 1.135–1.167× |
| 8192 | 4 | 50% | 0.16858 | 0.17529 | 0.958× | 0.955–0.962× |
| 8192 | 4 | 100% | 0.16860 | 0.17611 | 0.957× | 0.954–0.958× |
| 8192 | 128 | 0% | 1.02652 | 0.91480 | 1.095× | 1.036–1.150× |
| 8192 | 128 | 50% | 1.03167 | 1.03917 | 0.993× | 0.989–0.996× |
| 8192 | 128 | 100% | 1.03284 | 1.05234 | 0.991× | 0.902–0.994× |

All-rejected cases improved by **1.10–1.20×** in median paired speedup.
Hit-containing cases were approximately **1–6% slower** by median paired ratio;
individual seeds can regress more, as the ranges show. Prefer the default
unpruned path for hit-heavy or unknown workloads.

All 60 paired cases passed output-tolerance and acceptance-mask checks against
the unpruned path and output-tolerance checks against scalar reads. The maximum
observed difference from the unpruned batch output was
`5.960464477539063e-08`; this is measured floating-point agreement, not a claim
of bitwise equivalence for every possible input. Persistent state/archive
payload is identical for both variants.

## Reproduce

Use the project environment with Python 3.12 and NumPy 2.5.4 for the recorded
software configuration. The library itself retains its existing dependency
range.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m research.benchmark_sda_pruning
python -m unittest discover -s tests -v
python -m examples.quickstart
```

The tests cover float32/float64, gate boundaries, accepted/rejected ties across
archive tiles, retained keys, reservoir eviction, input validation, and the
elimination of irrelevant scalar fallbacks. All 45 tests and quickstart passed
locally before publication.

## Scope

This is a random-input NumPy implementation microbenchmark. It does not measure
LRA task accuracy, trained-model quality, write throughput, GPU performance, or
end-to-end inference. The exhaustive archive-search complexity is unchanged.
