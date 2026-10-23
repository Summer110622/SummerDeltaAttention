# Summer Delta Attention (SDA)

**Author: summer** · MIT License · research reference / unreviewed manuscript

Summer Delta Attention v5-Gated (SDA-5) combines a **no-decay delta-rule fast-weight state** with a **bounded archive of original key/value pairs** and a **confidence-gated residual correction**. This repository is an educational research prototype, not a pretrained LLM or optimized linear-attention kernel.

## Algorithm

For normalized keys `k` and queries `q`, with `F` of shape `(d_key, d_value)`:

```text
error = v - F.T @ k
F     = F + beta * outer(k, error)              # no multiplicative decay
j     = argmax_i cosine(q, archived_key[i])     # exact archive scan
accept = cosine(q, archived_key[j]) >= tau
out    = F.T @ q + accept * (archived_value[j] - F.T @ archived_key[j])
```

The archive is maintained by uniform reservoir sampling with a fixed maximum capacity. **It is not a constant-memory exact-recall solution:** it incurs `O(B(d_key+d_value))` archive memory and `O(B*d_key)` query-search work, in addition to the `O(d_key*d_value)` delta-state update. Exact retrieval of an archived key algebraically recovers its saved value, but this does not generalize to unseen or missing keys.

## Quick start

```bash
python -m pip install -e .
python -m examples.quickstart
python -m unittest discover -s tests -v
```

Python >=3.10 and NumPy >=1.24 are required. The [reference module](summer_delta_attention/reference.py) exposes `SummerDeltaAttention`, `kda_proxy_state` and `full_kv_attention`.

## Experiment figures

The following PNG plots are included in this repository under [`assets/readme/`](assets/readme/). They are derived from the **16-seed synthetic benchmark**, not language-model training. Reproduce the figures using `research/summary_results.csv` and `research/figures.py`.

### Main comparison: SDA vs KDA

<p align="center">
  <img src="assets/readme/fig_main_comparison.png" alt="Main comparison chart between SDA and KDA across methods and memory settings" width="900" />
</p>

**Figure 1.** In the synthetic exact-key recall benchmark, calibrated SDA-5 with a 50% archive reaches **0.4721 ± 0.0099** MSE, compared with **0.9336 ± 0.0098** for the KDA-style recurrent proxy. The README figure highlights the central caveat: SDA improves recall here by using a substantially larger external archive.

### Memory / accuracy Pareto view

<p align="center">
  <img src="assets/readme/fig_budget_pareto.png" alt="Pareto chart showing memory-accuracy tradeoff for SDA, KDA and baselines" width="980" />
</p>

**Figure 2.** This plot visualizes the trade-off between value-reconstruction error and total state-plus-storage footprint. KDA and no-decay DeltaNet remain compact (**4 KiB**) but have higher reconstruction error, whereas SDA lowers error by spending additional archive memory.

### Noise and sequence-length scaling

<p align="center">
  <img src="assets/readme/fig_noise_scaling.png" alt="Noise and sequence-length scaling figure for SDA, KDA and baselines" width="980" />
</p>

**Figure 3.** The third figure summarizes sensitivity to query noise and scaling with sequence length under the study protocol in `research/experiment.py`. These are controlled synthetic experiments, not downstream language-model evaluations.

## Manuscripts / 論文

- [English academic paper (PDF)](papers/Summer_Delta_Attention_vs_KDA_EN.pdf) · [English Markdown](papers/SDA_vs_KDA_EN.md)
- [日本語論文 (PDF)](papers/Summer_Delta_Attention_vs_KDA_JA.pdf) · [日本語Markdown](papers/SDA_vs_KDA_JA.md)
- [Prior numerical study and reproduction script](research/experiment.py) (NumPy and pandas; its reported results use **16 held-out random seeds**, separate development calibration, and an untrained channel-gated KDA recurrence **proxy**)

The included 5-page bilingual PDF papers contain the academic figures and full study; the Markdown versions give GitHub-readable equations, protocol, key measurements, and citations.

## Synthetic recall results (not a language-model benchmark)

| Method | Value MSE (n=512, exact-key queries) | State + stored KV (KiB) |
| --- | ---: | ---: |
| SDA-5 calibrated, 50% original KV archived | **0.4721 ± 0.0099** | 100 |
| Indexed KV, zero fallback, same archive | 0.4962 ± 0.0101 | 96 |
| KDA-style channel-decay proxy (untrained) | 0.9336 ± 0.0098 | **4** |
| No-decay DeltaNet, calibrated | 0.9301 ± 0.0098 | **4** |

These values come from the **separate benchmark** in `research/experiment.py`, not from `examples/quickstart.py`. KDA uses a deterministic development-calibrated gate, **not** a trained Kimi Linear implementation. SDA uses 25× the nominal state+KV capacity of the proxy. The sharp full-KV softmax comparator is not an equal-memory or trained full Transformer. The observed result **does not prove SDA is superior to KDA** or that the full hybrid is linear-time.

## Reproduce prior synthetic study

```bash
python -m pip install numpy pandas matplotlib
OPENBLAS_NUM_THREADS=1 python research/experiment.py
OPENBLAS_NUM_THREADS=1 python research/figures.py
```

See [study protocol](papers/SDA_vs_KDA_EN.md) for the dev/test split, confidence gate, fairness limitations and references. Reports and implementations are experimental and should be cited as unreviewed work.
