# eqmem_hard_v2 結果: SDA-Recency 再Go判定 — **Go（Go-1 通過）**

n=2048, ρ=0.5, B=1024, D=32, V=16, hard=rewrite+distractor, TEST 16 seeds, NumPy f64 CPU。Expanded β=0.4（calibのみ、既存と同値）。

| σ | 手法 | MSE ± SEM | KiB | rewrite_hit | stable_hit |
|---|---|---|---|---|---|
| 0 | **SDA-Recency（新）** | **0.575 ± 0.020** | 388 | 0.875 | 0.487 |
| 0 | Recency KV | 0.651 ± 0.024 | 384 | 0.875 | 0.487 |
| 0 | Expanded DeltaNet | 0.686 ± 0.009 | 388 | 0 | 0 |
| 0 | SDA-reservoir（旧） | 0.848 ± 0.019 | 388 | 0.486 | 0.488 |
| 0 | Indexed KV | 0.850 ± 0.019 | 384 | 0.486 | 0.488 |
| 0.4 | **SDA-Recency（新）** | **0.585 ± 0.019** | 388 | 0.005* | 0.003* |
| 0.4 | Recency KV | 0.654 ± 0.022 | 384 | 0.869 | 0.483 |
| 0.4 | Expanded DeltaNet | 0.915 ± 0.011 | 388 | 0 | 0 |
| 0.4 | SDA-reservoir（旧） | 0.855 ± 0.018 | 388 | 0.001* | 0.004* |

paired Δ（他手法 − 新核、95% CI, df=15）:
- σ=0: Expanded **+0.111 [+0.071, +0.151]** / Recency +0.076 [+0.061, +0.092] / 旧SDA +0.273 [+0.212, +0.335]
- σ=0.4: Expanded +0.331 [+0.287, +0.375] / Recency +0.069 [+0.056, +0.082]

## 判定
- **Go-1 通過**: Expanded − SDA-Recency の CI 下限 +0.071 > 0（σ=0）。σ=0.4 でも優位。
- rewrite_hit（σ=0）は Recency KV と一致（0.875）。メモリは 388 vs 388 KiB（±5% 内）。
- Recency KV にも両σで有意に勝つ（delta 状態 F が archive 外キーを補う分）。

## 正直な注意
- 変更は archive 選定のみ（β/τ 再サーチなし）。改善の主因は「rewrite 後の最新値保持」で、データ生成（distractor・rewrite が後半に来る）が recency に有利な構造である点に注意。一様過去クエリや固定B長系列（Data Synth案3）では未検証。
- \* σ=0.4 の rewrite_hit は「per-query MSE<1e-3」判定のため、ノイズ下で F 補正が残差を含む SDA 系はほぼ 0 になる。MSE で見るべき。
- 合成タスクのみ。LM 性能の主張はしない。
