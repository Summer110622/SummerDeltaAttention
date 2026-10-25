# ρ ablation memo (OFAT) — CALIB exploratory

One factor was varied: the recency ring width `archive_cap`. This page records that CPU float32 grid. It is exploratory CALIB only. It is not an LM result, and it is not a ranking of faces. Do not copy the cells below into the README, the papers, or `RESEARCH_STATUS.md`.

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
| τ | 0.75 at init, not swept and not re-locked. AdamW may move the knob |
| Gate | soft while training, `g̃ = σ((cos − τ) / T)`; hard `1[cos ≥ τ]` at eval. No STE |
| Optimizer | AdamW on `beta_logit`, `tau_logit`, `gate_log_temp` only. Two steps, no early stop, lr 3e-4, weight decay 0 |
| Memory | byte count is `entries * dtype.itemsize`. No hardcoded `* 8` |
| Shape | frozen Core stack: d_model 256, 6 layers, 4 heads, d_ff 1024, vocab 8000 |
| Train mix | Ordinary MQAR, same batch builder as `a3_cpu_smoke.py`: uniform pair index, batch 2, 20 pairs, 4 queries, length 47. The query token copies that pair's key. CPU float32, one thread |
| Eval mix | CALIB eval only. π_exact = 1, σ = 0. Every query token is its source key. The source is drawn from pairs still inside cap 8 when any exist, and that sequence is shared by all three cells. There is no noisy-query channel |
| Seeds | CALIB only. Train 20250001–20250004, eval 20250005–20250008, disjoint. Init seed is the train seed and is shared across the three caps. TEST, HOLDOUT, and LM/MQAR `20300001–` were not read |

One factor moved: ρ / `archive_cap`. No ρ×τ, ρ×β, or ρ×layout interaction. This round does not open τ. τ then β stay later, and only after a round where the hard gate is open and next-token CE is identifiable. That order is not a license to lower τ on this page.

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

MQAR length is 47. Caps 8 and 32 are shorter than that prefix, so those rings evict (`ring_evicts` true). Cap 96 is wider than the prefix, so the large ring does not evict: every past position stays inside the band. KiB still follows the nominal cap, because the byte rule counts `archive_cap` slots rather than the positions this length actually keeps. The byte rule on this run is 480, 768, and 1536 KiB (state sizes 491520, 786432, and 1572864 bytes). That is the cap check. It is not a quality score.

## Exact-key eval

The previous CALIB grid copied keys at a uniform pair index. Its `accepted_rate` was 0 on every cell, so a shared next-token CE could not be read as a ρ result.

This round keeps that train schedule and changes CALIB eval only.

* π_exact = 1 and σ = 0. The query token is the source key. No noise is added.
* The source is chosen from pair keys whose distance to the query is strictly inside cap 8, when that set is non-empty. Cap 8 is the smallest ring, so a source inside it is inside cap 32 and cap 96 as well. The three cells therefore see one eval sequence per seed.
* On this length the last query sits 8 steps after the final pair key. Cap 8's band is strict `< 8`, so that query has no in-ring source at the small cell. It still copies that nearest key (σ = 0). Caps 32 and 96 include distance 8.

`source_in_ring_rate` on the four eval seeds is 0.75 at cap 8 and 1 at caps 32 and 96. The jsonl field `which` is the same three times inside each train/eval pair.

## Critic health gate

Read this before next-token CE. The rule in the log is: each cell `accepted_rate` > 0, and the preferred bar is every CALIB eval seed > 0 and mean ≥ 0.05. `accepted_rate` is the mean over layers of that block's `acceptance_rate` (best in-ring cosine, including the current position, clears that block's τ).

| Cell | archive_cap | accepted_rate | every eval seed > 0 | mean ≥ 0.05 | max best in-ring cosine | lowest τ |
|---|---:|---:|---|---|---:|---:|
| small | 8 | 0 | no | no | 0.5177628397941589 | 0.7497373819351196 |
| base | 32 | 0 | no | no | 0.5728760361671448 | 0.7497373819351196 |
| large | 96 | 0 | no | no | 0.5743206143379211 | 0.7497373819351196 |

Every layer on every pair is 0 (`accepted_rate_per_layer` in the jsonl). The preferred bar fails with the cell bar. Per-seed rates are `[0, 0, 0, 0]` on each cell. τ was not lowered. The json `tau_mean` values sit within about 5e-5 of 0.75. The highest best-in-ring cosine on any of these forwards is the large-cell figure above, and it stays under the lowest τ.

An exact token copy is not a cosine of 1. The block scores `cos(W_q h, W_k h)`. Those are different projections, and on this frozen QKV stack the best in-ring cosine stayed below τ, including at positions whose source key is still in the ring.

## Label

**UNIDENTIFIABLE.** The hard gate never opened, so the archive correction is multiplied by zero at eval and `archive_cap` cannot move the hard-eval readout through that correction. Next-token CE, hit, and the internal MSE are stored in the json (`ce_interpretable` is false). This page does not read them, and it does not call a match across cells a null effect of ρ.

The follow-up in the log is `stopped_for_human_review`. τ stays 0.75. This run does not re-lock it and does not start another grid.

Soft-gate train losses can still differ by cap. Those two-step traces are `train_ce` in the jsonl. They are not the eval result.

## Non-claims

* No language-model efficacy. Two knob steps on a synthetic MQAR batch are not a trained layer.
* No 「全面勝ち」. Hard stays SDA-Recency and iid / old-skew stay Expanded on the face table. This grid does not reopen that table.
* No ρ main effect, and no claim that ρ has no main effect. The gate was shut, so CE is unidentifiable.
* No ρ×τ or ρ×β result. τ and β were not swept. τ was not re-locked after the exact-key mix.
* TEST 20261023–20261038 and HOLDOUT 20270001–20270008 stay sealed. The LM band was not mixed into these pairs.
* Do not promote a cap from KiB, from `source_in_ring_rate`, or from the stored hit rate.
