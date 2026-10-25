# ρ ablation memo (OFAT) — CALIB exploratory

One factor was varied: the recency ring width `archive_cap`. This page records that CPU float32 grid. It is not an LM result, and it is not a ranking of faces. Do not copy the cells below into the README, the papers, or `RESEARCH_STATUS.md`.

The command re-checks the float16 dual-write gate and does not score a cell unless that gate returns exit 0. This run's log records that check as PASS on CALIB seeds 20250001–20250008. Rules for the gate stay in [`EVAL_PROTOCOL_JA.md`](EVAL_PROTOCOL_JA.md) and in the constants at the top of `gate_f16_dualwrite.py`.

```bash
python research/sdalm_10m/ablation_rho.py
```

Logs: `logs/ablation_rho_calib.json` (means) and `logs/ablation_rho_calib.jsonl` (one row per cell and CALIB pair).

## What stays fixed

| Covariate | Value for every cell |
|---|---|
| Layout | SDA-Recency only. Expanded and Hybrid are not in this grid |
| Archive policy | The scaffold ring: the last `archive_cap` positions. DESIGN.md's continuous-key stand-in for Recency-B. This run does not add a same-key overwrite test |
| β | 0.05 at init, not swept. AdamW may move the knob |
| τ | 0.75 at init, not swept. AdamW may move the knob |
| Gate | soft while training, `g̃ = σ((cos − τ) / T)`; hard `1[cos ≥ τ]` at eval. No STE |
| Optimizer | AdamW on `beta_logit`, `tau_logit`, `gate_log_temp` only. Two steps, no early stop, lr 3e-4, weight decay 0 |
| Memory | byte count is `entries * dtype.itemsize`. No hardcoded `* 8` |
| Shape | frozen Core stack: d_model 256, 6 layers, 4 heads, d_ff 1024, vocab 8000 |
| Task | MQAR hard, same batch as `a3_cpu_smoke.py`: batch 2, 20 pairs, 4 queries, length 47. CPU float32, one thread |
| Seeds | CALIB only. Train 20250001–20250004, eval 20250005–20250008, disjoint. Init seed is the train seed and is shared across the three caps. TEST, HOLDOUT, and LM/MQAR `20300001–` were not read |

One factor moved: ρ / `archive_cap`. No ρ×τ, ρ×β, or ρ×layout interaction. If a later round sees a main effect of ρ on the protocol's next-token CE, the next single factor is τ, and only after that β. That order is τ then β. It is not a license to open the grid now.

## Horizon C

`DESIGN.md` and `EVAL_PROTOCOL_JA.md` name `C` as the horizon a run multiplies. They do not assign an integer.

An earlier round set **C = 32** and took large = `round(0.75 * 32) = 24`. That ring is narrower than the Core cap 32, so the three budgets were not monotonic. This round does not use 24.

This round sets **C = 128 for the large cell only**. Base stays the Core point. Small stays the previously agreed low cell.

| Cell | How the cap is set | archive_cap |
|---|---|---|
| small | Explicit low cell **8**. `round(0.25 * 128) = 32`, which collides with the Core cap, so small is not `0.25 * C`. `round(0.25 * C) = 8` only when C = 32, and 32 is not the horizon used for large | **8** |
| base | Core budget point. Not `round(0.5 * 128)` (that product is 64, and 64 is not a cell) | **32** |
| large | `round(0.75 * 128) = 96`. The product is not clipped down to the Core cap | **96** |

The caps increase: **8 < 32 < 96**. Large is 96, which is at least 64 and wider than the Core point. ρ = 0.75 is the large cell's fraction of C. Small and base are not fractions of this C (`fraction_of_C` is false on those rows; large stores ρ = 0.75).

MQAR length is 47. Caps 8 and 32 are shorter than that prefix, so those rings evict (`ring_evicts` true). Cap 96 is wider than the prefix, so the large ring does not evict: every past position stays inside the band. `source_in_ring_rate` is 1 on every CALIB pair at cap 96. KiB still follows the nominal cap, because the byte rule counts `archive_cap` slots rather than the positions this length actually keeps.

## The three points

Means over the four CALIB pairs. Next-token CE is the full-sequence hard-eval loss, the same reduction as `a3_cpu_smoke.py`. Perplexity is `exp` of that mean. MSE is the mean, over MQAR answer positions, of `Σ_v (softmax(logit)_v − 1_{v = target})^2` on this cell's own logits. It is an internal Brier score, not the synthetic-face value MSE. `hit_rate` is this cell's own past-ring argmax: the fraction of query positions whose strictly earlier in-ring argmax token equals the query token, averaged over heads and layers. The current position is not counted; it is the query, not a retrieved write. `accepted_rate` is the mean over layers of that block's `acceptance_rate` (best in-ring cosine, including the current position, clears that block's τ). KiB is one sequence, six layers, float32 item size.

| Cell | Cap rule | archive_cap | next-token CE | MSE / ppl | hit_rate | accepted_rate | KiB |
|---|---|---:|---:|---:|---:|---:|---:|
| small | explicit 8, not 0.25×C | 8 | 9.063178777694702 | 0.999904990196228 / 8631.544853171987 | 0.020833333333333336 | 0 | 480 |
| base | Core point | 32 | 9.063178777694702 | 0.999904990196228 / 8631.544853171987 | 0.016927083333333332 | 0 | 768 |
| large | 0.75 of C=128 | 96 | 9.063178777694702 | 0.999904990196228 / 8631.544853171987 | 0.016927083333333332 | 0 | 1536 |

`logs/ablation_rho_calib.json` field `eval_ce_max_abs_diff_across_cells` is 0. The three CE figures are the same value, not a rounded tie. The same identity holds for the answer-position MSE. Sample sd of CE across the four pairs is about 0.0086 inside every cell. Per-pair rows are in the jsonl.

KiB is the byte rule at these caps: 480, 768, and 1536. It grows with `archive_cap` (small < base < large) and does not depend on the batch. State sizes are 491520, 786432, and 1572864 bytes.

`accepted_rate` is 0 on every layer and every pair. τ stayed near the 0.75 init (the jsonl `tau_mean` values sit within about 1e-4 of 0.75). The hard gate never opened, so the archive correction was multiplied by zero at eval. Under that schedule, `archive_cap` cannot move next-token CE. The match is that fact. It is not evidence that the rings are interchangeable once the gate opens, and it is not an LM measurement.

`hit_rate` does move between cap 8 and the wider rings. The means are 0.020833 (cap 8), 0.016927 (cap 32), and 0.016927 (cap 96). Sample sds (n = 4) are 0.015333, 0.0078125, and 0.008895. The cap-32 and cap-96 means are the same value; the four pairs are not the same series (jsonl `hit_rate`). The spread is on the same scale as the gap from cap 8. QKV is not in the AdamW group, so this rate is the frozen random projections plus the ring, not a trained retriever. Separately, the fraction of queries whose source key is still inside the ring — geometry, not the mask — is 0.125, 0.71875, and 1 at caps 8, 32, and 96. The large rate is 1 because no query-to-source distance on this length-47 batch reaches 96. Wider rings keep more source keys. They also keep more other positions. The argmax-hit mean did not rise from cap 32 to cap 96. That is a description of this CALIB run, not a choice of winner.

The soft-gate training loss does differ by cap inside a seed. Those two-step traces are `train_ce` in the jsonl. They are not the eval table.

## Non-claims

* No language-model efficacy. Two knob steps on a synthetic MQAR batch are not a trained layer.
* No 「全面勝ち」. Hard stays SDA-Recency and iid / old-skew stay Expanded on the face table. This grid does not reopen that table.
* No ρ×τ or ρ×β result. τ and β were not swept.
* TEST 20261023–20261038 and HOLDOUT 20270001–20270008 stay sealed. The LM band was not mixed into these pairs.
* Do not read the shared CE as a quality score, and do not promote the hit-rate order into a selected cap.
