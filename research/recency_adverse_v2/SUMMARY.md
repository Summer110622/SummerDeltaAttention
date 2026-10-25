# M-D3b v2 recency不利ストレス（recency-adverse-stress-v2, query_mode=old_skew_pi, π=0）

実行: `OPENBLAS_NUM_THREADS=1 .venv/bin/python research/recency_adverse_v2/run_adverse_v2.py`（約3秒）。eqmem_hard.py / experiment.py / v1 は非改変・v1再ランなし。
クエリ ids ~ Unif{0..N−B−1}（`rng(seed+9_000_000)`、キー/値/z/perm は v1 と同じ `make_data`）。TEST 16 seed、SDA β=0.05・τ=0.75 凍結（assert）。Expanded β は同じ old-skew 分布の CALIB seed で格子選択。
コード内 assert: query_mode=='old_skew_pi'（uniform_past拒否）、π=0、CALIB∩TEST=∅、β/τ凍結、健全性FAILで例外終了。

## 1. 健全性: **PASS**（sanity.csv）
| N,B | in_B(recency) | out_B(recency) | in_B(一様B) | B/N |
|---|---|---|---|---|
| 512,128 | 0.000 | 1.000 | 0.236 | 0.250 |
| 512,256 | 0.000 | 1.000 | 0.501 | 0.500 |
| 2048,128 | 0.000 | 1.000 | 0.065 | 0.0625 |
| 2048,256 | 0.000 | 1.000 | 0.120 | 0.125 |

## 2. 不利の確認（Goセル N=512,B=128,σ=0,π=0,TEST16）: **PASS**
- MSE(recency窓SDA − 一様B SDA) = **+0.231 [+0.203, +0.258]**（CIが0を含まない）
- Indexed KV も同方向: **+0.239 [+0.211, +0.267]**
- MSE: SDA一様B 0.716 / SDA recency 0.946（= delta単独 0.946、archiveは全く当たらない）/ Indexed一様 0.748 / Indexed recency 0.986 / Expanded 0.645
- 他3セルも全て同方向で有意（+0.470, +0.069, +0.124）。

## 3. SDA-Recency vs Expanded（等メモリ、Δ=Expanded−SDA-Recency、正ならSDA勝ち）
| N,B | σ=0 | σ=0.4 |
|---|---|---|
| 512,128 | −0.301 [−0.313,−0.289] | −0.137 [−0.152,−0.121] |
| 512,256 | −0.382 [−0.397,−0.367] | −0.183 [−0.198,−0.168] |
| 2048,128 | −0.119 [−0.125,−0.112] | −0.042 [−0.051,−0.032] |
| 2048,256 | −0.188 [−0.196,−0.179] | −0.084 [−0.093,−0.075] |
→ 不利条件では SDA-Recency は全8セル（σ=0.4含む）で Expanded に有意に負ける。単一F補正は寄与ゼロ（No-decay DeltaNet と同値）。

## 判定
- v2 仕様は有効（健全性PASS・不利確認PASS）。
- SDA-Recency 単独は old-skew で全敗 → 主核としては不適格。B外の圧縮補完が必須。

ファイル: run_adverse_v2.py, raw_results.csv, summary_results.csv, paired_results.csv, sanity.csv, configuration.json, run.log
