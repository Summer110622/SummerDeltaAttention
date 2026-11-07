# SDA-SoftChunk: reduced real-data encoder pilot

This note documents the prototype primitive and an exploratory comparison run.
**It is not an official Long Range Arena score or evidence of general model
superiority.** The benchmark used small, previously inspected train/validation/
test subsets and 3 seeds; no untouched confirmatory holdout remains.

## Prototype primitive

For normalized input keys and values, the chunkwise update solves, per chunk,

`(I + beta * tril(K K^T, -1)) U = beta * (V - K S_in)`

then updates `S_out = S_in + K^T U`. It is algebraically equivalent to the
sequential no-decay delta update, subject to floating-point evaluation order.
`SoftChunkMixer` retains at most `capacity` input positions via Algorithm R and
reads a softmax mixture of their original KV pairs. Its output is

`q S + g(q) * (soft_archive(q) - soft_archive_keys(q) S)`.

The temperature and sigmoid gate are learned. There is no straight-through
gradient through discrete top-1 choice, archive sampling, or padding masks.
Set a `valid_mask` to keep padded positions out of writes and retrieval.

For the reported run, beta was 0.05, chunk and capacity were 64, initial gate
was 0.8, initial inverse temperature was 4. A parallel `archive_soft` ablation
used the same 64 archived entries but no delta state/gate.

## Exploratory results

3 seeds, 256 steps/model, width 32, one encoder layer, CPU FP32, 1 thread. Accuracy
is mean ± seed-to-seed SD, not a confidence interval. Latency is p50 over timed
model forwards; each cell shows SDA-SoftChunk / KDA recurrence encoder.

| Task | Test accuracy, SoftChunk / KDA | batch 1 p50, ms | batch 16 p50, ms | batch 16 RSS, MiB |
|---|---:|---:|---:|---:|
| Original author-published ListOps | 21.50% ± 2.26pp / 18.40% ± 1.23pp | 1.003 / 9.047 | 5.815 / 17.055 | 333.0 / 325.4 |
| Grayscale CIFAR-10 pixel sequence | 14.53% ± 1.03pp / 14.07% ± 0.42pp | 1.417 / 17.737 | 13.065 / 38.544 | 332.8 / 340.8 |

The predeclared point-estimate target against this KDA encoder was met in all
four task/batch combinations: mean accuracy no lower, p50/p95 lower, and median
RSS no more than 5% higher. This is a small-pilot engineering check, not
statistical dominance. **The candidate is not the overall winner:** its latency
still trails Full and Linear attention; on ListOps the archive-only ablation
scored 22.10%, above SoftChunk's 21.50%; on the image task it scored 12.07%.
The former means this pilot does not isolate a benefit from the delta state or
gate. Previous SDA also scored higher than SoftChunk on image accuracy (15.33%
vs 14.53%).

For fairness, the benchmark applied the same exact CLS-only readout/MLP pruning
to all six methods, then reverified stored accuracies for 36 checkpoints. It
timed token IDs-to-logits, including fresh delta state construction, projections
and classification head; text preprocessing, file I/O and model loading were
excluded. All were CPU reference implementations. RSS is worker-process VmHWM,
including framework, test inputs and allocator, not activation-only or KV-only
memory. Full conditions, seed ranges, and caveats are in the raw experimental
measurements (`run 8b96826d` in the originating research workspace).

## Scope distinctions

- The KDA comparator is a learned-gate, noncausal final-state recurrence
  encoder. It omits Kimi Linear's complete layer design, short convolution,
  gated normalization/output path, hybrid schedule, and pretrained weights.
- ListOps data comes from the original authors, with long expressions filtered
  to 512 tokens. It is not the LRA Long ListOps release.
- CIFAR-10 is a reduced grayscale 1,024-pixel sequence pilot.
- Results do not measure GPU fused kernels, decoding, or all six LRA tasks.

The optional `trainable` dependency extra is required to import this PyTorch
prototype; the NumPy streaming package remains independent of PyTorch.
