# 今日タスク1: 同メモリ hard スイート 結果（2026-10-25 JST）

実装: `research/eqmem_hard.py`（`experiment.py` の seed 分割・`state_update`・`paired_vs_sda`・committed ハイパー流用。既存CSV非破壊）
再現: `OPENBLAS_NUM_THREADS=1 .venv/bin/python research/eqmem_hard.py --n 2048 --rho 0.5`（約3秒）
出力: `summary_results.csv` / `paired_results.csv` / `rewrite_hit.csv` / `raw_results.csv` / `configuration.json`

## 設定
- n=2048, D=32, V=16, Q=128, ρ=0.5 → B=1024。test 16 seeds（df=15）、calib 8 seeds。holdout 未使用
- hard: ユニーク key U=⌊0.85n⌋=1740（うち128本はディストラクタ）、残り308トークンは既存 key の上書き。各クエリ正解 key に 15–30° の hard negative を最終書込より後に1本挿入。クエリの半分は上書きされた key。正解は最新 value
- SDA/Indexed のアーカイブは位置の一様 reservoir（同一 key 重複時は最新位置を優先）。SDA β=0.05, τ=0.75 は committed のまま
- Expanded DeltaNet: R+1=97 個の D×V ブロック、固定ランダム射影 argmax で key をルーティング。β のみ calib（calib グリッドで β=0.4 を選択）
- Recency KV: 直近 B 個の distinct key を最新 value で保持（LRU-B）、閾値なし top1
- メモリ: SDA = 388.0 KiB（プロトコル本文の 384.5 は計算違い。(512+1024·48)·8/1024=388）。Expanded 388.0 KiB（±0%）、Recency/Indexed 384.0 KiB

## 主表（σ=0, MSE mean±SEM）
| 手法 | MSE | KiB |
|---|---|---|
| Recency KV (eq-B) | 0.651±0.024 | 384 |
| Expanded DeltaNet (eq-mem) | 0.686±0.009 | 388 |
| SDA v5-Gated (calibrated) | 0.848±0.019 | 388 |
| Indexed KV (reservoir) | 0.850±0.019 | 384 |
| No-decay DeltaNet | 1.003±0.009 | 4 |
| KDA proxy | 1.003±0.009 | 4 |

paired ΔMSE（comparator − SDA、95% CI）: Expanded −0.162 [−0.205, −0.120] / Recency −0.197 [−0.268, −0.126] / Indexed +0.0016 [−0.0003, +0.0034] / Delta +0.155 / KDA +0.155

rewrite_hit（上書きクエリ、per-query MSE<1e-3）: Recency 0.875 / SDA 0.486 / Indexed 0.486 / Expanded・Delta・KDA 0

## Go / No-go
1. **SDA ≻ Expanded（CI下限>0）: 失敗**（Expanded の方が有意に低MSE）
2. **SDA vs Recency: 失敗**（Recency が有意に良く、rewrite_hit も SDA より高い）
3. 回帰なし vs Indexed（CI下限≥−0.005）: **合格**（下限 −0.0003。ただし差はほぼゼロで、SDA の delta 部の寄与はこの設定では実質なし）
4. Expanded メモリ ±5%: **合格**（388.0 vs 388.0 KiB）

**判定: No-go**（Go-1 失敗）。プロトコルに従い Roadmap へ設計差し戻し。

## 解釈（正直な注記）
- n=2048 では一様 reservoir が正解位置の約49%しか保持せず、SDA は Indexed とほぼ同じ挙動。no-decay delta 状態（D×V=512）は1740 key で飽和し補正にほとんど寄与しない
- 同メモリなら、状態をハッシュ分割した Expanded DeltaNet（1ブロックあたり平均約21 key）の方が容量効率が高い
- Recency の優位は一部データ設計由来: 上書きで key が新しくなるため、上書きクエリ（半数）が FIFO に残りやすい。非上書きクエリでは Recency の stable_mse 1.05 で SDA（0.79）より悪い。クエリ分布を変えると順位は変わり得る
- 付録 σ=0.4: Expanded 0.915 で SDA 0.855 が逆転（ΔCI [+0.025,+0.097]）。Recency は依然 0.654 で最良。SDA の rewrite_hit は σ=0.4 で厳密閾値 1e-3 を満たさない（fast state の残差）ため 0.001 と出るが MSE は Indexed と同等
- 合成 exact-key 連想想起のみ。LM 性能に関する主張はしない
