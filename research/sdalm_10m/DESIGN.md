# SDALM-10M design

CPU PyTorch scaffold for a causal stack at the shape frozen by Core A1. It is a place to run a smoke step. It is not evidence that a trained layer works, and it does not update the synthetic face table.

## Frozen shape

| | |
|---|---|
| variant | **SDA-Recency** (default). Expanded is `variant="expanded"` only |
| d_model | 256 |
| layers | 6 |
| heads | 4 (d_head = 64) |
| d_ff | 1024 |
| vocab | 8000 |
| archive | base ring `archive_cap=32` |
| LM head | separate from the token embedding |
| positions | none. Order is the causal scan. `max_seq_len=128` is a runtime cap, not a parameter width |

These widths are not retuned to force a round 10,000,000. The learned-parameter total is whatever this shape produces (printed by `model.py`). It lands inside 1e7 ± 20%. Tying the LM head would drop one `8000 × 256` table and fall outside that band, so the head stays untied.

SDA knobs that train are β, τ, and the gate temperature, plus the usual embeddings, QKV, output projection, FFN, and RMSNorm scales. The lift used by Expanded is not learned. Archive ρ is not learned.

## Face-wise results stay face-wise

Equal-memory synthetic recall (σ=0), not this model. Sources: `BEST_RESULTS.md`, `papers/SDA5_M-L1_limitations_JA.md`.

| Face | Best layout | Where |
|---|---|---|
| hard (rewrite + distractor) | **SDA-Recency** | `research/eqmem_hard_v2/` |
| iid (uniform past) | **Expanded DeltaNet** | `research/hybrid_rf/` |
| old-skew | **Expanded DeltaNet** | `research/recency_adverse_v2/` |

Hard-face SDA-Recency at MSE 0.575 ± 0.020 is the hard face, not an overall best. iid and old-skew stay with Expanded. H′, Hybrid-RF, and Cond2 were No-go (`RESEARCH_STATUS.md`). This file does not merge the two winners.

## Why Recency vs Expanded is a config

They differ in where one budget of state lives.

* **SDA-Recency** keeps a no-decay delta matrix and a bounded archive of recent pairs, then applies the gated correction from `summer_delta_attention/reference.py`. On the hard face the archive policy that won was Recency-B, with β = 0.05 and τ = 0.75 left frozen (`research/eqmem_hard_v2/PROTOCOL.md`).
* **Expanded** spends that budget on a wider delta state and has no exact archive. It is not "Recency plus a bigger state."

`SDALMConfig.variant` selects one layout for the whole stack.

* `recency` — state `(d_head, d_head)` per head, ring of the last `archive_cap` writes, soft gate in train mode.
* `expanded` — no ring and no τ. Keys go through a fixed lift (`(-1)^i * roll` on the head axis, then re-normalized). Default `expand_factor` is the smallest integer whose entries are at least `d_head² + 2 * archive_cap * d_head`. At 64 and 32, both sides have 8192 entries per head. `expand_factor` is rejected on the recency config.

A forward never runs the ring and the lift together. Nothing switches layout from token statistics.

## Soft gate, and the hard gate as an eval switch

Training always uses the soft correction

```text
fast = Sᵀ q
α    = softmax(cos(q, K_ring) / temperature)
g    = sigmoid((α · cos − τ) / temperature)
out  = fast + g (α V − Sᵀ (α K))
```

`hard_gate_eval` replaces α and g with argmax and `1[max cos ≥ τ]` only when the module is in `eval()`. Training ignores that flag so the hard threshold cannot cut the gradient. β and τ start at 0.05 and 0.75 (the reference SDA-Recency init, and the frozen hard-face settings). Smoke training may move them. Expanded has no τ. Its β starts at the same `beta_init` because this scaffold does not reload per-face NumPy calibration and does not treat that calibration as an LM hyperparameter. The gate temperature starts at 0.25 so the sigmoid is not flat on the first step. It is a softness knob on the same recency correction.

Two gaps versus the NumPy kernel, so this is not sold as a port:

* Synthetic SDA-Recency keeps distinct key ids, latest write wins. Here keys are continuous, so the block keeps the last `archive_cap` positions. No reservoir sample, no extra "same key" test.
* The NumPy class loops in Python. This block does not. See below.

## Hot path

`S_t = (I − β k_t k_tᵀ) S_{t−1} + β k_t v_tᵀ` is an affine recurrence. The implementation scans it with batched matmuls whose offset doubles each step (Hillis–Steele). The Python loop is over those offsets, `log2(T)` times, not over tokens. The ring is one `(T, T)` causal band. Both are differentiable.

This is not the chunkwise WY algorithm in DeltaNet, not Gated DeltaNet, and not Kimi Delta Attention. There is no `fla` dependency. Workspace for the scan is proportional to `T · d_head²` per head, and the band is proportional to `T²`. That workspace is not the memory rule below, and it is not a claim of linear-time training.

## Memory and rates

Per-step state entries, per head:

* recency: `d_head² + archive_cap · (d_head + d_head)`
* expanded: `expand_factor · d_head²`

Bytes are `entries * dtype.itemsize` via `torch.empty((), dtype=dtype).element_size()`. The code does not multiply by 8. A float32 model and a float64 count of the same entries differ. `rule_state_nbytes` is this per-step rule summed over layers and the batch. It is not the scan workspace. At float32, the six-layer per-step rule state for one sequence is 786,432 bytes (`8192 × 4 heads × 6 layers × 4`). It does not grow with length. Float64 of the same entries is twice that, because the item size changed.

`accepted_rate` is the fraction of positions whose best in-ring cosine clears τ, computed from that recency block's own scores. Expanded's diagnostic dict does not contain `accepted_rate`. Neither variant emits `hit_rate` (there is no key-id match to count). Do not copy one method's rate onto another. The legacy writer in `research/experiment.py` currently stores the calibrated-SDA hit and accept on every method row; this scaffold does not repeat that.

## Archive ρ later

The OFAT plan is [`ABLATION_RHO.md`](ABLATION_RHO.md). It is not a run. ρ ∈ {small ≈ 0.25, base = archive_cap 32, large ≈ 0.75 or the Core cap}. β stays 0.05, the gate stays soft in train and hard in eval, and the archive stays Recency-B. τ then β are touched only after ρ shows a main effect. The table stays empty until `gate_f16_dualwrite.py` exits 0.

## Parameter count

Untied embeddings, biases off, no position table. Recency:

| Piece | Count |
|---|---:|
| Token embedding `8000 × 256` | 2,048,000 |
| LM head `256 × 8000` | 2,048,000 |
| 6 layers: QKV+out `4 × 256²`, FFN `2 × 256 × 1024`, two RMSNorms, β/τ/temperature (`3 × 4`) | 4,721,736 |
| Final RMSNorm | 256 |
| **Total** | **8,817,992** |

Expanded drops τ and the temperature (`6 × 4 × 2 = 48`) and adds no lift parameters: **8,817,944**. `python research/sdalm_10m/model.py` prints the live count. If the frozen shape's wiring changes, update `DEFAULT_*_PARAMS` in `train_smoke.py` in the same commit.

## Citations for a later linear-attention comparison

Context only. No number from these papers is a result of this repository, and this code does not implement their kernels.

* Schlag, Irie, Schmidhuber (2021). Linear Transformers Are Secretly Fast Weight Programmers. ICML. (delta rule / DeltaNet)
* Yang, Wang, Zhang, Shen, Kim (2024). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. NeurIPS. arXiv:2406.06484. (parallel DeltaNet)
* Yang, Kautz, Hatamizadeh (2025). Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR. arXiv:2412.06464. (Gated DeltaNet)
* Kimi Team (2025). Kimi Linear: An Expressive, Efficient Attention Architecture. arXiv:2510.26692. (Kimi Delta Attention / KDA)

## Limitations

* Do not claim language-model efficacy, and do not claim a trained layer works. The smoke loss is one batch.
* No universal equal-memory kernel. No full-attention SOTA, and no SOTA of any other kind.
* The hard face is not the overall best. Hard = SDA-Recency. iid and old-skew = Expanded.
* Do not describe this stack with 「有望な中間」 or 「全面勝ち」.
* Do not cite the old README gap (SDA 0.4721 vs a 4 KiB KDA-style proxy 0.9336) as a win. That pair is unequal memory, and the proxy is not Kimi Linear.
* Archive ρ ∈ {small, base, large} is the first ablation later. It is not in this run.
