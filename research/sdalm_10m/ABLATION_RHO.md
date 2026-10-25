# ρ ablation memo (OFAT) — not run

Archive retention is the first factor to vary, and only that factor. This page is the plan. It is not a result table. Do not fill the metric cells from a partial run, a smoke loss, or another method's rates.

The float16 dual-write gate has to exit 0 before any cell below is launched:

```bash
python research/sdalm_10m/gate_f16_dualwrite.py
```

Rules for that gate are locked in [`EVAL_PROTOCOL_JA.md`](EVAL_PROTOCOL_JA.md) and in the constants at the top of `gate_f16_dualwrite.py`.

## What stays fixed

| Covariate | Value for every cell |
|---|---|
| Layout | SDA-Recency only. Expanded and Hybrid are not in this grid |
| Archive policy | Recency-B (latest write of a near-duplicate key; otherwise the newest ring) |
| β | 0.05, not swept |
| τ | 0.75 at the start of a future run, not swept in this round |
| Gate | soft while training, `g̃ = σ((cos − τ) / T)`; hard `1[cos ≥ τ]` at eval. No STE |
| Memory | byte count is `entries * dtype.itemsize`. No hardcoded `* 8` |
| Shape | frozen Core stack: d_model 256, 6 layers, 4 heads, d_ff 1024, vocab 8000 |

One factor moves: ρ. No ρ×τ, ρ×β, or ρ×layout interaction in this round. If a later round sees a main effect of ρ, the next single factor is τ, and only after that β. That order is τ then β. It is not a license to open the grid now.

## The three points

`archive_cap = 32` is the Core budget point. It is the base cell, not a fitted fraction. `C` is the horizon a future run will multiply. `C` is not chosen here, so small and large do not get invented integer caps in this memo.

| Cell | ρ | Archive budget | MSE / ppl | hit_rate | accepted_rate | KiB |
|---|---|---|---|---|---|---|
| small | ≈ 0.25 | `round(0.25 * C)` once `C` is fixed. Not chosen in this memo | not run | not run | not run | not run |
| base | Core budget point | **archive_cap = 32** | not run | not run | not run | not run |
| large | ≈ 0.75, or the Core cap | `round(0.75 * C)`, clipped to the Core cap if that product would exceed it. The Core cap for the base cell is 32; a larger hard max is not set here | not run | not run | not run | not run |

`hit_rate` and `accepted_rate` are computed inside the method that produced the masks. A Recency cell does not donate its rates to another method. KiB uses that cell's dtype item size.

Empty cells are intentional. Launching the grid before the float16 gate passes is out of protocol.
