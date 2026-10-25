# 研究状況（RESEARCH_STATUS）— 2026-10-25（Asia/Tokyo）

正本: [`papers/SDA5_M-L1_limitations_JA.md`](papers/SDA5_M-L1_limitations_JA.md) ・ 再現: [`research/face_repro_ml2/`](research/face_repro_ml2/README.md)

## 現状（一段落）

合成連想記憶タスク（未学習 NumPy・等メモリ帯・σ=0・TEST 16 seed）で比較した結果、**単一アーカイブ核の優劣は面（データ／クエリ分布）に依存する**ことが分かった。hard 面（上書き＋妨害）では SDA-Recency が最良だが、一様過去（iid）と古いクエリ寄り（old-skew）では Expanded DeltaNet が最良で、SDA-Recency は下位になる。3面を同時に制する核を狙った H′ / Hybrid-RF / Cond2 はいずれも No-go で、**等メモリの万能核は実証失敗**である。現在は新規アーキ実験と LM 学習を止め、公開用の整理段階にある。

## 面別 SOTA（σ=0・等メモリ帯）

| 面 | 最良 | MSE（代表、低いほど良い） | 証跡 |
|---|---|---|---|
| hard（rewrite+distractor, n=2048） | **SDA-Recency** | **0.575 ± 0.020**（次点 Recency KV 0.651、Expanded 0.686）。Δ(Expanded−SDA-Recency)=+0.111 [+0.071, +0.151] | `research/eqmem_hard_v2/` |
| iid 一様過去 | **Expanded DeltaNet** | 512/128: Expanded 0.594 / SDA-Recency 0.680 | `research/hybrid_rf/` |
| old-skew（π=0） | **Expanded DeltaNet** | 512/128: Expanded 0.645 / SDA-Recency 0.946 | `research/recency_adverse_v2/` |

注: old-skew の N512/B256 では SDA-uniform（0.479）が Expanded（0.567）より低いが、**非推奨・holdout 必須**で面の推奨核にはしない（数字の単独引用禁止）。

## 失敗核（主核不適格）

| 核 | 判定 | 要点 |
|---|---|---|
| H（hard 過大主張） | 主核不適格 | hard の勝ちは recency 有利な生成に依存。old-skew で Recency 全敗。全体最良へ外挿しない |
| H′（SDA-Hyper-EqMem, C=4） | No-go | MSE 0.750、Expanded に有意負け。**仕様は C=4 No-go。C=16 は test 閲覧後の事後アブレーションで holdout 未確認のため採用しない・主結果にしない** |
| Hybrid-RF（配分A） | No-go | σ=0 全9セルで Expanded 未達。hard 0.741 |
| Cond2（hybrid_switch） | No-go | hard で誤 Exact、rewrite_hit 0.371≪0.875、MSE≈0.993 |

## 再現コマンド1枚（正本: [`research/face_repro_ml2/COMMANDS.md`](research/face_repro_ml2/COMMANDS.md)）

```bash
cd SummerDeltaAttention
python -m pip install -e ".[research]"
export OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python   # なければ python3

# --- 成功面 ---
$PY research/eqmem_hard_v2/run_v2.py              # hard → SDA-Recency
$PY research/recency_adverse_v2/run_adverse_v2.py # adverse → Expanded
$PY research/hybrid_rf/run_hybrid.py              # iid 表出典（＋ Hybrid-RF 失敗証跡）

# --- 失敗証跡 ---
$PY research/hybrid_switch/run_cond2.py           # Cond2 No-go
$PY research/eqmem_hard_hyper/run.py              # H′ Hyper-EqMem No-go

# --- 面別表 ---
$PY research/face_repro_ml2/build_tables.py
# または成功面のみ: bash research/face_repro_ml2/reproduce.sh
```

表: `research/face_repro_ml2/table_hard.csv` / `table_iid.csv` / `table_adverse_v2.csv`  
詳細サマリ: 各ディレクトリの `SUMMARY.md`。

## 主張しないこと（明示）

- **LM での有効性は一切主張しない**（LM・学習済み層・GPU・速度は未測定）。
- **等メモリの万能核は未達**。「全面で勝つ」「万能」「KDA に勝った」とは言わない。
- README の旧 KDA 代理比較（0.4721 vs 0.9336）はメモリ約25倍差の不均等比較であり、主結果ではない。

## 次の予定

C（公開用整理）→ 必要なら A（LM/学習タスクへ戦場変更）。現時点で新規アーキ実験・LM 学習は停止中。

## 未コミット成果物の扱い

`research/eqmem_hard*`・`hybrid_*`・`recency_adverse*`・`face_repro_ml2/`・本ファイル・`papers/SDA5_M-L1_limitations_JA.md` は**証跡としてリポジトリ内に残す**。公開 PR で main に載せる範囲は README の研究状況節・`RESEARCH_STATUS.md`・M-L1・face_repro を優先し、巨大 raw CSV は必要に応じて `.gitignore` または LFS 方針を別途決める（現時点では削除しない）。
