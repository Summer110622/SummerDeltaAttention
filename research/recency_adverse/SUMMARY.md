# M-D3b recency不利ストレス（recency-adverse-stress-v1）結果

実行: `OPENBLAS_NUM_THREADS=1 .venv/bin/python research/recency_adverse/run_adverse.py`（約3秒）。eqmem_hard.py / experiment.py は非改変。
TEST 16 seed（20261023–038）、SDA β=0.05・τ=0.75 は凍結。Expanded DeltaNet の β だけ CALIB seed で格子選択（N512B128=0.4, N512B256=0.7, N2048=0.2）。HOLD 未使用。
データ: `make_data`（i.i.d. キー、rewrite・distractorなし）、クエリ ids~Unif{0..N-1}。Expanded は SDA と同KiB（52 / 100 KiB）。

## 1. 健全性（P(out_B) ≈ 1−B/N ±5pt）: **PASS**
| N,B | 理論 | recency窓 | 一様B |
|---|---|---|---|
| 512,128 | 0.750 | 0.726 | 0.762 |
| 512,256 | 0.500 | 0.477 | 0.503 |
| 2048,128 | 0.938 | 0.945 | 0.946 |
| 2048,256 | 0.875 | 0.880 | 0.880 |

## 2. 不利の確認（N=512,B=128,σ=0 で recency窓SDA が一様B SDA より有意に悪い）: **FAIL（不利は再現せず）**
一様B − recency = **+0.030 [+0.007, +0.053]** → 向きは逆で、recency窓の方がむしろ良い。
理由: キーもクエリも i.i.d. なので、「末尾B」と「一様B」はどちらも任意のB個の部分集合で、期待カバー率は同じ（B/N）。差は seed ごとのカバー率の揺らぎ（recency 27.4% vs 一様 23.8%）だけで説明がつく（in_B MSE はどちらも0、out_B MSE ≈ delta単独）。N=2048 では差がゼロをまたぐ（−0.001, −0.007）。
→ **この仕様は構造的に recency 不利を作れない**。不利を作るには、クエリを古い側に寄せる（例: age ∝ 一様で B 外を過半にする、または古いキーほどクエリされやすくする）か、rewrite／時間で変わる分布が必要。仕様 v2 は Data Synth に差し戻すべき。

## 3. Roadmap主問: SDA-Recency は Expanded に勝ち続けるか → **σ=0 では負け、σ=0.4 ではおおむね勝ち**
paired Δ = Expanded − SDA-Recency（正ならSDA-Recencyの勝ち）
| N,B | σ=0 | σ=0.4 |
|---|---|---|
| 512,128 | **−0.086** [−0.107,−0.066] | +0.095 [+0.069,+0.121] |
| 512,256 | **−0.031** [−0.052,−0.009] | +0.319 [+0.295,+0.342] |
| 2048,128 | **−0.075** [−0.090,−0.061] | +0.005 [−0.010,+0.020]（差なし） |
| 2048,256 | **−0.081** [−0.096,−0.066] | +0.026 [+0.008,+0.045] |

MSE（σ=0）: N512B128 Expanded 0.594 / SDA-Recency 0.680 / SDA一様 0.710 / Indexed(rec) 0.710 / DeltaNet 0.930 / Zero 0.979。
解釈: 一様過去クエリで正解の大半がB外になると、exact archive はB内しか救えず、B外は単一F（実質飽和）頼みになる。Expanded は全キーを圧縮保持するので、σ=0 の B外MSE が低い（例 N512B128: 0.645 vs 0.938）。σ=0.4 ではExpandedの圧縮読出がノイズで崩れ、SDA側が勝つ。
→ **eqmem_hard_v2 の「SDA-Recency ≻ Expanded」（σ=0）は、このストレス条件では残らない**。勝因は recency 選択そのものより「hard データのクエリがB内に偏っていた」ことの寄与が大きい可能性。

## 4. その他
- SDA の delta 補正は Indexed より小さく有意に良い（N=512 で +0.015〜+0.030）。N=2048 ではゼロ（F 飽和）。
- 主張範囲: 合成・i.i.d.・NumPy float64のみ。LM/latency 主張なし。

## 判定まとめ
- 健全性: PASS
- 不利の確認: FAIL（仕様上起きない。仕様の見直しが必要）
- 主核問（σ=0 で SDA-Recency ≻ Expanded）: **No**。全4セルで Expanded が有意に優位。σ=0.4 では SDA-Recency が3/4セルで優位、1セルは差なし。
- 推奨: M-D4 の核凍結は保留。「B外キーの圧縮補完」（例: F を Expanded型の分割状態にした SDA-Recency + Expanded 併用）を次候補として Arch に戻す。

ファイル: run_adverse.py, raw_results.csv, summary_results.csv, paired_results.csv, configuration.json, run.log
