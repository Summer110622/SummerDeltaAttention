# SDA-RC: Summer Delta Attention + reservoir computing (PyTorch prototype)

Trainable language-model prototype combining:

- **Frozen reservoir** (`reservoir.py`): diagonal complex linear echo-state network,
  |λ| log-spaced in [0.5, 0.995], evaluated in parallel by FFT convolution, with
  NG-RC style fixed quadratic features. Only the linear readout is trained.
- **Trainable SDA layer** (`sda_layer.py`): the SDA-5 rule from
  `summer_delta_attention/reference.py` with learned q/k/v/β/τ —
  no-decay delta state plus a gated top-1 archive correction
  `o = Sᵀ(q − g·k*) + g·v*`. Archive membership is an exact Algorithm-R
  reservoir sample, so a memory budget can be enforced. `archive="soft"` uses a
  softmax over the archive; `archive="none"` is plain DeltaNet.
- `model.py` wires them into pre-norm blocks (reservoir readout conditions the
  mixer's q/k/v) and provides baselines: `sda`, `sda_soft`, `delta`, `attn`,
  `reservoir`.

```bash
pip install torch numpy
python -m unittest tests.test_sda_lm -v
python -m sda_lm.train --task mqar --mixer sda --steps 1500
python -m sda_lm.train --task lm --mixer sda --corpus-glob '/path/to/text/**/*.txt'
```

## First results: MQAR (single seed — preliminary)

16 key–value pairs then 16 queries (64 tokens), 64 keys/64 values, chance ≈ 1.6%.
d_model 64, 2 layers, 2 heads, 1500 steps × batch 32, CPU. Raw data:
`results/mqar.jsonl`.

| Mixer | Reservoir | KV budget | Query acc | Loss | Trained params |
| --- | --- | --- | ---: | ---: | ---: |
| Softmax attention | yes | all | **0.525** | **1.50** | 148k |
| Softmax attention | yes | 16 | 0.275 | 3.14 | 148k |
| SDA, soft archive | yes | all | 0.247 | 2.36 | 132k |
| DeltaNet (no archive) | yes | — | 0.213 | 2.60 | 132k |
| SDA, hard archive | yes | 16 | 0.202 | 2.63 | 132k |
| SDA, hard archive | yes | all | 0.189 | 2.73 | 132k |
| Softmax attention | no | all | 0.115 | 2.87 | 156k |
| SDA, hard archive | no | all | 0.104 | 2.88 | 83k |
| Reservoir readout only | yes | — | 0.016 | 4.16 | 91k |

What this does and does not show:

- **The frozen reservoir helped every model it was added to** (SDA 0.104 → 0.189,
  attention 0.115 → 0.525), at zero trained recurrent parameters. On its own
  it cannot do recall (chance level), as expected for a fading-memory system.
- **The hard-gated archive did not help yet**: SDA-hard (0.189) ≤ DeltaNet
  without archive (0.213); the soft archive is slightly better (0.247). The
  straight-through top-1 gives no gradient to *which* key is chosen, which is the
  likely culprit. Attention with full KV is clearly best.
- With a 16-token budget, SDA (0.202) keeps its accuracy while budgeted
  attention drops to 0.275 with a much worse loss — the bounded-memory regime is
  where SDA is closest, but it still does not win.
- One seed, short training, nothing near convergence, tiny models. These are
  smoke-test numbers, not evidence of superiority over anything.
