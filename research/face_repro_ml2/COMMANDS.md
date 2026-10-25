# 再現コマンド1枚（M-L2 / 公開整理 C）

日付: 2026-10-25 Asia/Tokyo  
新実験なし。既存スクリプトの再実行のみ。正本は本ファイル。要約はリポジトリ直下 [`RESEARCH_STATUS.md`](../../RESEARCH_STATUS.md)。

```bash
cd /workspace/sda-lab/SummerDeltaAttention   # またはクローン先ルート
python -m pip install -e ".[research]"
export OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python   # なければ python3
```

## 成功面（面別 SOTA 証跡）

| 面 | 勝者 | ディレクトリ | コマンド |
| --- | --- | --- | --- |
| **hard** | SDA-Recency | `research/eqmem_hard_v2/` | `$PY research/eqmem_hard_v2/run_v2.py` |
| **adverse** (old-skew) | Expanded | `research/recency_adverse_v2/` | `$PY research/recency_adverse_v2/run_adverse_v2.py` |
| **iid**（表の出典） | Expanded | `research/hybrid_rf/`（iid suite） | `$PY research/hybrid_rf/run_hybrid.py` |

面別表の再集計のみ（ラン不要）:

```bash
$PY research/face_repro_ml2/build_tables.py
# → table_hard.csv / table_iid.csv / table_adverse_v2.csv
```

成功面をまとめて再ラン:

```bash
bash research/face_repro_ml2/reproduce.sh
```

## 失敗証跡（No-go / 主核不適格）

| 核 | 判定 | ディレクトリ | コマンド |
| --- | --- | --- | --- |
| Hybrid-RF（配分A） | No-go | `research/hybrid_rf/` | `$PY research/hybrid_rf/run_hybrid.py` |
| Cond2（切替） | No-go | `research/hybrid_switch/` | `$PY research/hybrid_switch/run_cond2.py` |
| H′ Hyper-EqMem | No-go（仕様C4） | `research/eqmem_hard_hyper/` | `$PY research/eqmem_hard_hyper/run.py` |

各ディレクトリの `SUMMARY.md` + `paired_results.csv` が判定根拠。

## 一括（成功面 → 失敗証跡 → 表）

```bash
export OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python
# 成功面
$PY research/eqmem_hard_v2/run_v2.py
$PY research/recency_adverse_v2/run_adverse_v2.py
# iid 数値 + Hybrid-RF 失敗証跡（同一ラン）
$PY research/hybrid_rf/run_hybrid.py
# その他失敗証跡
$PY research/hybrid_switch/run_cond2.py
$PY research/eqmem_hard_hyper/run.py
# 面別表
$PY research/face_repro_ml2/build_tables.py
```

## Seed（共通）

CALIB `20250001`–`08` / TEST `20261023`–`38` / HOLDOUT `20270001`–`08` 未使用。

## 主張しないこと

LM・GPU速度・等メモリ万能核。詳細は `RESEARCH_STATUS.md`。
