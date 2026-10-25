# eqmem_hard 再Go: SDA-Hyper-EqMem（Arch 案H）結果（2026-10-25 JST）

実装: `/workspace/sda-lab/sda_hyper/sda_hyper/hyper_eqmem.py`（`SDAHyperEqMem`）
ランナー: `research/eqmem_hard_hyper/run.py`（`research/eqmem_hard.py` を import のみ、非改変）
再現: `OPENBLAS_NUM_THREADS=1 .venv/bin/python research/eqmem_hard_hyper/run.py`（約12秒）
出力: `summary_results.csv` / `paired_results.csv` / `raw_results.csv` / `configuration.json`

## 設定
- eqmem_hard と同一データ（n=2048, D=32, V=16, Q=128, rewrite+15–30° distractor, test 16 seeds, calib 8 seeds, float64）
- 本案: no-decay delta F（常時）＋ 正確(k,v)プール P（同一key upsert・LRU退避）＋ LSH L=3, b=7, バケット容量C（満杯時は最古ポインタを落とす）＋ top2 cos の margin gate
- calib（CALIB seeds, σ∈{0,0.4}平均）グリッド β∈{0,0.05,0.2}, τ∈{.5,.65,.75,.85}, m∈{0,.02,.05,.1}
  - 両バリアントとも **β=0, τ=0.85, m=0** が選択 → **delta F と margin gate は寄与なし**（calibが無効化を選んだ）
- 比較: SDA / Expanded / Recency は eqmem_hard と同じ実装・設定（Expanded β=0.4, 97ブロック）

## メモリ
| 手法 | KiB |
|---|---|
| HyperEqMem 仕様（P990, C4）: F4 + pool371.25 + ptr6 + hyp5.25 | 386.5 |
| HyperEqMem アブレーション（P946, C16）: F4 + pool354.75 + ptr24 + hyp5.25 | 388.0 |
| Expanded / SDA | 388.0 |
| Recency | 384.0 |

## 主表（MSE mean±SEM, 16 seeds）
| 手法 | σ=0 | σ=0.4 | rewrite_hit σ=0 | stable_hit σ=0 |
|---|---|---|---|---|
| HyperEqMem 仕様 C4 | 0.750±0.010 | 0.885±0.012 | 0.647 | 0.276 |
| HyperEqMem C16（事後アブレーション） | **0.569±0.014** | 0.641±0.014 | 0.834 | 0.445 |
| Recency KV | 0.651±0.024 | 0.654±0.022 | 0.875 | 0.487 |
| Expanded DeltaNet | 0.686±0.009 | 0.915±0.011 | 0 | 0 |
| SDA v5 | 0.848±0.019 | 0.855±0.018 | 0.486 | 0.488 |

## paired ΔMSE（comparator − 本案, 95% CI, df=15）
| 本案 | vs | σ=0 | σ=0.4 |
|---|---|---|---|
| 仕様 C4 | Expanded | −0.064 [−0.092, −0.036] | +0.030 [−0.008, +0.068] |
| 仕様 C4 | Recency | −0.098 [−0.145, −0.052] | −0.231 |
| 仕様 C4 | SDA | +0.098 [+0.051, +0.145] | −0.031 |
| C16 | Expanded | **+0.117 [+0.082, +0.151]** | +0.275 [+0.233, +0.316] |
| C16 | Recency | +0.082 [+0.057, +0.108] | +0.013 [−0.023, +0.050] |
| C16 | SDA | +0.279 [+0.226, +0.333] | +0.214 |

## Go / No-go
- **仕様どおりの案H（C=4）: No-go**。Expanded の方が有意に良い（CI上限 −0.036）。SDA には勝つ。
- 原因: 128バケット×C=4 に対し保持key≈990（平均≈7.7/バケット）で、LSHポインタが半分以上落ち、プールに残っていても候補に出ない。設計の容量計算ミス。
- **C=16・P=946（同388 KiB）: Go-1 条件を満たす**（下限 +0.082 > 0）。Recency にも σ=0 で有意勝ち。
- ただし C16 は **test 結果を見た後に選んだ事後変更**であり、厳密には未確定。holdout（20270001–08）で凍結構成のまま確認するまで「暫定Go」扱いを推奨。

## 正直な注記
- 勝因はほぼ「同一key upsert + LRU の正確KVプール」で、Recency KV の改良版に近い。delta F（β=0選択）と margin gate（m=0選択）は効いていない。「SDA」要素の寄与はこの設定ではゼロ。
- Recency に勝つのは、閾値τ=0.85で外れを0出力（低MSE側）に落とせるため（stable_mse 0.87 vs Recency 1.05）と推測。
- LSH は近似のため hit は Recency より少し低い。読出は O(L·b·D + L·C·(D+V)) で n 非依存だが、Python実装で速度は未測定・未最適化。
- データ設計上、上書きクエリが半数で LRU 系に有利。クエリ分布が変われば順位は変わり得る。
