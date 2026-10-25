# SDA-NG

Working title. Synthetic recall only. This directory does not claim an LM result, a universal core, or a reproduction of the published hard-face TEST MSE.

Design freeze: [`DESIGN.md`](DESIGN.md).

## Face winners (unchanged)

| Face | Winner | Evidence |
| --- | --- | --- |
| hard (rewrite + distractor) | **SDA-Recency** — no-decay delta `F` + Recency-B archive | [`research/eqmem_hard_v2/`](../eqmem_hard_v2/) |
| iid (uniform past) | **Expanded DeltaNet** (eq-mem) | [`research/hybrid_rf/`](../hybrid_rf/) |
| old-skew (adverse_v2) | **Expanded DeltaNet** (eq-mem) | [`research/recency_adverse_v2/`](../recency_adverse_v2/) |

Hybrid-RF and Cond2 stay No-go. They are not revived here.

## Why the soft gate

SDA-Recency's hard accept is `g = 1[cos ≥ τ]` with frozen `τ = 0.75`. On the prior LM path that gate never opened: `accepted_rate = 0` because max cosine ≈ 0.574 even for exact-key queries (`π = 1`). With the gate stuck shut, a ρ cross-entropy on top of it is not identifiable.

SDA-NG keeps the hard-face skeleton and replaces only the accept indicator:

```text
cos_j = max_i ⟨q, k_i⟩
T     = max(temperature, ε)     # ε = 1e-6
g     = σ((cos_j − τ) / T)      # temperature = 0.10, τ = 0.75
out   = Fᵀq + g · (v_j − Fᵀk_j)
```

An optional CALIB-only `τ ← Quantile_{0.3}(cos)` is implemented and documented in `DESIGN.md`. It is an in-sample diagnostic on the smoke, not a TEST result. The margin gate (案2) is not implemented.

## Smoke

CALIB seeds `20250001`–`20250008` only. σ = 0 exact keys on the reconstructed hard stream. TEST 16-seed is not run.

```bash
OPENBLAS_NUM_THREADS=1 python research/sda_ng/smoke.py
```

Committed log: [`smoke_log.json`](smoke_log.json). `mean_g ≤ 0` exits with status `fail_gate_stuck_shut` and does not continue to TEST.

Frozen-τ result on that log (8 CALIB seeds, not TEST, not the published 0.575):

- soft `mean_g` = 0.7167, `frac(g > 0.05)` = 0.9941
- where `cos < 0.75` the hard weight is 0 and the soft `mean_g` is 0.1191
- MSE soft = 0.2735 vs hard-gate SDA-Recency = 0.2632 (soft − hard = +0.0104). On this exact-key face the soft gate is slightly worse: `g(cos=1) ≈ 0.924` shrinks exact recovery, and archive misses get a small pull toward the wrong neighbor.
- The in-sample CALIB quantile landed at τ ≈ 1 (point mass at cosine 1), so the hard-accept target of about 0.7 did not occur. That diagnostic is worse (soft MSE 0.4427) and is not a second tuned result.
