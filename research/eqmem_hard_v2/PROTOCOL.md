# 再Go判定・最小実験表（eqmem_hard_v2）

- 日付: 2026-10-25 JST
- 既存 `research/eqmem_hard.py` / `research/eqmem_hard/` は**変更禁止**（No-go 証跡）
- 本ラウンドの出力はこのディレクトリのみ

## 候補（1つだけ）

**SDA-Recency（新核）**  
アーカイブ選定のみ Algorithm R → Recency-B（LRU distinct、最新位置）。  
`F`・β=0.05・hard gate τ=0.75 は committed のまま。β/τ 再サーチなし。

ねらい: No-go 主因（rewrite で reservoir が古い値を残す）を最小変更で潰し、同メモリ Expanded を超える。

## 固定

| 項目 | 値 |
|---|---|
| n, ρ, B | 2048, 0.5, 1024 |
| D, V, Q | 32, 16, 128 |
| hard データ | `eqmem_hard.make_hard_data` と同一 |
| test / calib | 既存 TEST_SEEDS / CALIB_SEEDS |
| Expanded | R+1=97、β は calib のみ（既存どおり） |
| メモリ目標 | 新核 ≈ Expanded ≈ 388 KiB（±5%） |

## 最小実験表

| セル | 手法 | σ | 必須 | 仮説 |
|---|---|---|---|---|
| C0 | SDA-Recency（新核） | 0 | ✓ | rewrite_hit≈Recency。MSE≺Expanded |
| C1 | SDA-Recency（新核） | 0.4 | ✓ | ノイズ下でも Expanded 以下〜同等 |
| E0 | Expanded DeltaNet (eq-mem) | 0 | ✓ | 主比較。既報≈0.686 |
| E1 | Expanded DeltaNet (eq-mem) | 0.4 | ✓ | 既報≈0.915 |
| R0/R1 | Recency KV (eq-B) | 0 / 0.4 | 参照 | Go 式には入れない |
| S0/S1 | SDA-reservoir（旧核） | 0 / 0.4 | 参照 | 回帰チェック |

測るもの: MSE±SEM、memory KiB、rewrite_hit / stable_hit、rewrite_mse / stable_mse、paired Δ。

## Go / No-go

1. **Go-1（本体・σ=0）**: 新核 ≻ Expanded。`Δ = Expanded_MSE − 新核_MSE` の 95% Student-t CI **下限 > 0**（16 seed, df=15）。
2. **Go-1b**: σ=0.4 も同じ paired を報告（符号反転でも即 No-go にはしないが SUMMARY に明記）。
3. **Go-2（付帯）**: σ=0 で 新核 `rewrite_hit ≥ Recency_hit − 0.05`（実質 Recency 級）。
4. **Go-3（付帯）**: 新核 KiB が Expanded の ±5% 以内。
5. **No-go**: Go-1 失敗 → 設計差し戻し。本ディレクトリ結果は残す。

## 実装メモ（Experimenter 向け）

- 入口案: `research/eqmem_hard_v2/run.py`
- `eqmem_hard` の `make_hard_data` / `hard_queries` / `recency_positions` / `expanded_*` を import
- 新核の archive index = `recency_positions(key_id, B)`、読出は既存 SDA ゲート式
- 成果物: `configuration.json`, `raw_results.csv`, `summary_results.csv`, `paired_results.csv`, `rewrite_hit.csv`, `SUMMARY.md`
- 再現: `OPENBLAS_NUM_THREADS=1 python research/eqmem_hard_v2/run.py`

## Page

[再Go判定・最小実験表](https://cursor.com/canvas/store-46a8036e-bca4-43b1-aef7-e42194f8451f/d41db107-7ef7-4c37-9ab9-b90cd38518db)

## 実行
`OPENBLAS_NUM_THREADS=1 .venv/bin/python research/eqmem_hard_v2/run_v2.py`（eqmem_hard.py を import のみ、無改変）。出力: raw/summary/paired_results.csv, configuration.json, run.log, SUMMARY.md
