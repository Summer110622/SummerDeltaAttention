# M-L2 面別再現パッケージ（新規アーキなし）

日付: 2026-10-25 Asia/Tokyo  
担当: SDA Experimenter  
方針: 既存証跡のみ。新しいアーキテクチャ実装は含まない。面ごとに「その面の勝者」を固定し、再現コマンドと表を1か所に置く。

**再現コマンド1枚（成功面＋失敗証跡）: [`COMMANDS.md`](COMMANDS.md)**（[`RESEARCH_STATUS.md`](../../RESEARCH_STATUS.md) にも同内容を統合済み）。

## 面 → 勝者

| 面 | 勝者 | 証跡ディレクトリ | メモ |
| --- | --- | --- | --- |
| **hard**（rewrite+distractor, n=2048 ρ=0.5） | **SDA-Recency** | `research/eqmem_hard_v2/` | archive=Recency-B、β/τ凍結 |
| **iid**（一様過去クエリ） | **Expanded DeltaNet (eq-mem)** | `research/hybrid_rf/` の iid suite（数値は `recency_adverse` 系と同生成器） | 全4セルで Expanded≻SDA-Recency |
| **adverse_v2**（old-skew π=0） | **Expanded DeltaNet (eq-mem)** | `research/recency_adverse_v2/` | 対 SDA-Recency は全セル Expanded 優位 |

採用しない（参照のみ）: Hybrid-RF / Cond2（いずれも No-go）。仕様 H′ は **C=4 No-go**；C=16 は test 閲覧後・holdout 未確認のため**採用しない／本再現パックに含めない**。旧 reservoir SDA も hard 以外の主推奨にしない。

## 主表（σ=0, TEST 16 seeds, mean MSE）

### hard（N=2048, B=1024, KiB≈388）

> hard の Recency 優位は当該生成（rewrite が後半）に依存。old-skew では不成立。

| 手法 | MSE | rewrite_hit |
| --- | ---: | ---: |
| **SDA-Recency（勝者）** | **0.575 ± 0.020** | 0.875 |
| Recency KV | 0.651 ± 0.024 | 0.875 |
| Expanded (eq-mem) | 0.686 ± 0.009 | 0 |
| SDA-reservoir（旧） | 0.848 ± 0.019 | 0.486 |

paired Δ = Expanded − SDA-Recency: **+0.111 [+0.071, +0.151]**（SDA-Recency 優位）

### iid（一様過去）— 勝者 Expanded

| N | B | Expanded | SDA-Recency | Δ Exp−Recency [95%CI] |
| ---: | ---: | ---: | ---: | --- |
| 512 | 128 | **0.594** | 0.680 | −0.086 [−0.107, −0.066] |
| 512 | 256 | **0.415** | 0.445 | −0.031 [−0.052, −0.009] |
| 2048 | 128 | **0.881** | 0.956 | −0.075 [−0.090, −0.061] |
| 2048 | 256 | **0.812** | 0.893 | −0.081 [−0.095, −0.066] |

出典 CSV: `table_iid.csv`（`hybrid_rf` raw から再集計）

### adverse_v2（old-skew）— 勝者 Expanded（対 Recency）

| N | B | Expanded | SDA-Recency | SDA-uniform | Δ Exp−Recency [95%CI] |
| ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 128 | **0.645** | 0.946 | 0.716 | −0.301 [−0.313, −0.289] |
| 512 | 256 | 0.567 | 0.948 | 0.479 | −0.382 [−0.397, −0.367] |
| 2048 | 128 | **0.872** | 0.991 | 0.922 | −0.119 [−0.125, −0.112] |
| 2048 | 256 | **0.813** | 1.001 | 0.877 | −0.188 [−0.196, −0.179] |

注: N512 B256 の SDA-uniform 0.479 は **非推奨・holdout 必須**（面の推奨核にしない・単独引用禁止）。面の「推奨核」は Expanded（対 SDA-Recency）。

## 再現コマンド（新規アーキなし）

リポジトリルート: `SummerDeltaAttention/`

```bash
cd /workspace/sda-lab/SummerDeltaAttention
# 依存（既存どおり）
python -m pip install -e ".[research]"

# 1) hard → SDA-Recency 証跡を再生成
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/eqmem_hard_v2/run_v2.py

# 2) adverse_v2 → Expanded 証跡を再生成
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/recency_adverse_v2/run_adverse_v2.py

# 3) iid 面の数値（Expanded 勝者）— hybrid_rf ランに iid suite が含まれる
#    ※ Hybrid 自体の結果は No-go・採用しない。iid 数値抽出のみに使う
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/hybrid_rf/run_hybrid.py

# 本パッケージの表だけ再集計（ラン不要、既存 CSV から）
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/face_repro_ml2/build_tables.py
```

ワンショット: `bash research/face_repro_ml2/reproduce.sh`（上記1–3＋表再集計。所要は環境依存、目安数十秒）。

## Seed / 漏洩ルール（共通）

- CALIB: `20250001`–`08` / TEST: `20261023`–`38` / HOLDOUT `20270001`–`08` は未使用
- hard / adverse の SDA β=0.05, τ=0.75 は committed 凍結（面で再サーチしない）
- Expanded β は各スイートの CALIB のみ

## このパッケージに含めないもの

- Hybrid-RF / Cond2 の再実装や追加スイープ
- LM プロキシ・latency 主張
- 新規アーキ（XP 等）

## ファイル

| ファイル | 内容 |
| --- | --- |
| `README.md` | 本説明（表＋コマンド） |
| `table_hard.csv` / `table_iid.csv` / `table_adverse_v2.csv` | 機械可読な面別表 |
| `build_tables.py` | 既存 raw から表を再生成 |
| `reproduce.sh` | 既存ラン＋表再生成 |
