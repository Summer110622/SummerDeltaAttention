# Best face-wise results (equal-memory, synthetic recall)

Not an LM benchmark. No universal kernel.

| Face | Best | Representative MSE (σ=0) | Evidence |
|---|---|---:|---|
| hard (rewrite+distractor, n=2048 B=1024) | **SDA-Recency** | **0.575 ± 0.020** | `research/eqmem_hard_v2/` |
| iid (uniform past) | **Expanded DeltaNet** | e.g. 0.415 (N512 B256) | `research/hybrid_rf/` (iid suite) |
| old-skew (adverse_v2) | **Expanded DeltaNet** | e.g. 0.645 (N512 B128) | `research/recency_adverse_v2/` |

Reproduce: `research/face_repro_ml2/COMMANDS.md`.

Do not claim LM performance, universal SOTA, or adopt Hybrid/Cond2/H′/C16.
