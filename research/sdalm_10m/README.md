# SDALM-10M scaffold

Causal language-model stack at the frozen shape **d_model=256, 6 layers, 4 heads, d_ff=1024, vocab 8000**. PyTorch, CPU. Default mixer is SDA-Recency. Expanded is a second config, not the default and not a merge.

This directory is a scaffold plus a smoke train. It does not show that a trained layer works.

Design notes: [DESIGN.md](DESIGN.md). ρ plan (not run): [ABLATION_RHO.md](ABLATION_RHO.md). A3 protocol (canonical): [EVAL_PROTOCOL_JA.md](EVAL_PROTOCOL_JA.md). Frozen face-wise table: [`BEST_RESULTS.md`](../../BEST_RESULTS.md).

## Run

From the repository root, Python ≥ 3.10:

```bash
python -m pip install torch
# A CPU wheel is enough. CUDA is not required.
# Or: python -m pip install -e ".[sdalm]"

python research/sdalm_10m/model.py
python research/sdalm_10m/train_smoke.py
```

`model.py` prints the parameter count for recency and for `variant=expanded`, plus per-step state bytes at the parameter dtype (item size, not a fixed 8).

`train_smoke.py` uses seed 0, CPU, and four AdamW steps on one fixed synthetic MQAR batch. AdamW updates only `beta_logit`, `tau_logit`, and `gate_log_temp`. Other parameters keep gradients for the finite-grad check and are not in the optimizer. The script also checks the frozen parameter count, that the vectorized delta scan matches a sequential reference, and that acceptance is defined only on the recency block. The loss lines are that smoke. They are not an evaluation.

`python research/sdalm_10m/gate_f16_dualwrite.py` compares a float64 reference with a float16 dual-write on synthetic KV from CALIB seeds 20250001–20250008 only. Exit 0 is PASS. It does not read LM batches from 20300001 onward. The ρ table is not a run; it waits on that gate.

```bash
python research/sdalm_10m/train_smoke.py --variant expanded
python research/sdalm_10m/train_smoke.py --steps 4 --seed 0
```

β starts at 0.05. Recency τ starts at 0.75. Both are learnable, as is the gate temperature. The forward uses a soft gate. A hard gate runs only in eval when `hard_gate_eval` is set. Archive ρ is not trained; a later ablation is ρ ∈ {small, base, large} around the base ring of 32.

The batch in `data_mqar.py` is random integer ids. No corpus is downloaded. The mixer does not loop over tokens in Python: the delta rule is an associative scan, and the ring is a banded score tensor. That is not a fused kernel and not `fla`.

## Limitations

* No claim that the model, or any trained layer, is effective at language modeling. Not TinyStories. Not a SOTA result.
* Not equal to full attention. The stack has no softmax attention baseline.
* No universal equal-memory kernel. Hard-face Recency is not the overall best: hard → SDA-Recency, iid and old-skew → Expanded. Hybrid-RF, Cond2, and H′ stay No-go.
* Recency and Expanded are two configs. The forward does not mix the ring with the widened state.
* Do not use the phrases 「有望な中間」 or 「全面勝ち」 for this stack.
* Do not cite the repository's older SDA-vs-proxy table as a win over KDA. That comparison is unequal memory, and the proxy is not Kimi Linear.
* No `fla`, and no Kimi Delta Attention dependency. Papers a later comparison would cite (DeltaNet, Gated DeltaNet, Kimi Linear / KDA) are listed in DESIGN.md. Their numbers are not results of this scaffold.
* `accepted_rate` is recency-only, from that block's own scores. Expanded does not get a copied rate. This scaffold does not emit `hit_rate`.
* The loss lines from `train_smoke.py` stay out of papers, the repo-root README, and `RESEARCH_STATUS.md`.
