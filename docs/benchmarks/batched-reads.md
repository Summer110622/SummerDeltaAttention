# Tiled batch reads: CPU benchmark

`SummerDeltaAttention.read_batch` batches independent reads of a fixed state.
The comparison is against `np.stack([memory.read(q) for q in queries])` on the
same populated object and inputs. The public scalar `read`/`write` are unchanged.

## Observed results

| Archive capacity | Scalar ms / 256 queries | Batch ms / 256 queries | Paired speedup |
|---:|---:|---:|---:|
| 512 | 5.058369 | 0.409591 | 12.349805× |
| 4096 | 9.659748 | 1.770341 | 5.570457× |

Each latency is the median of three seed-level medians, each from seven repeats.
Speedup is the median of the three seed-level latency ratios, **not** the ratio
of the two aggregate latencies. All six numerical comparisons passed
`assert_allclose(rtol=2e-5, atol=2e-5)`; maximum absolute difference was
`4.76837158203125e-7`. This is tolerance-based agreement, not bitwise equality.

| Capacity | Seed | Scalar median ms | Batch median ms | Speedup |
|---:|---:|---:|---:|---:|
| 512 | 0 | 5.058369000835228 | 0.40959099897008855 | 12.34980508251996 |
| 512 | 1 | 5.2447600010054884 | 0.470274999315734 | 11.152538426743481 |
| 512 | 2 | 4.832022999835317 | 0.38832099926366936 | 12.443372902824606 |
| 4096 | 0 | 10.630034001223976 | 1.8639280006027548 | 5.703028227370611 |
| 4096 | 1 | 9.5077219993982 | 1.7068119996110909 | 5.570456501105336 |
| 4096 | 2 | 9.659748000558466 | 1.7703409994282993 | 5.456433536633853 |

## Hardware, software and protocol

- AMD Ryzen 5 3600, Linux x86-64, Python 3.12.12, NumPy 2.5.4.
- OpenBLAS 0.3.34.106.0, Haswell runtime dispatch, one thread verified at runtime
  with NumPy's runtime report (threadpoolctl 3.6.0 was installed for introspection).
- `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`.
- `key_dim=value_dim=64`, model dtype float32, capacities 512 and 4096.
- Random generator seeds 0, 1, 2. Independent normal keys and values; keys are
  normalized by `write`. All written keys fit in the archive.
- 256 queries: 128 sampled written keys and 128 independent normal vectors.
  The generated query array is float64; both read paths cast internally.
- `beta=0.05`, `threshold=0.75`, `overwrite_tol=1e-9`.
- `batch_size=256`, `archive_block_size=4096`, `details=False`.
- Each path is called once for numerical checking/warmup, then seven times
  for timing, alternating which path runs first on successive repetitions.
- Timing includes normalization, search, correction and output allocation.
  Input generation, model construction and all writes are outside the timer.
- Measurements were taken on the implementation in this PR before publication;
  the benchmark's orchestration-specific metadata was removed for this public
  version, without changing its workload, timing loop or read implementation.

## Reproduce

After the normal project installation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m research.benchmark_reads
```

The script prints effective settings, Python/NumPy/BLAS metadata, all individual
timings, per-seed medians, numerical differences and aggregate summaries.
The library still only requires NumPy. `threadpoolctl` is optional for richer
BLAS runtime reporting; it is not needed to run the benchmark.

## Limits

This is a single-machine, short, warm-cache synthetic microbenchmark. Three
data seeds are not independent machines and no confidence interval is claimed.
Performance depends on BLAS, dimensions, query count and archive size. These
results do not measure single-query latency, online write/read sequences,
writes, float64 performance, SDA+, GPU kernels or language-model performance.
The tested capacities fit within one archive tile; multiple archive tiles are
covered by correctness tests, but their performance was not measured here.

Every archive entry is still scanned: search remains O(Q * B * d_key).
The score tile is at most `batch_size * archive_block_size` floating scalars;
at defaults it uses 4 MiB in float32. Other temporary arrays, input/output,
diagnostics and BLAS-internal buffers are additional. Numerical near-ties and
gate boundaries use scalar fallbacks and can reduce speedup.
