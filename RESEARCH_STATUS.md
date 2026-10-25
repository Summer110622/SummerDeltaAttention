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

C（公開用整理）完了（GitHub `v2` / PR #7）。**Path A（LM/学習）着手中**（2026-10-25）。
新規単一核アーキ実験は当面停止。LM 着手の初期スコープは下記「A 初期タスク」参照。未測定のまま LM 性能は主張しない。

### A 今週タスク（Roadmap 確定・依存 A1→A2→A3）
1. **A1 差し込み設計（CPU境界）** — Transformer1層・学習可能は β/τ/gate のみ。面別2系統（SDA-Recency / Expanded）並記。
2. **A2 最小学習ループ（CPU・PyTorch）** — 1バッチ再現＋全パラメータ勾配有限。性能比較は出さない。
3. **A3 評価プロトコル草案＋CPUスモーク** — softmax MHA / no-decay Delta、seed・漏洩防止。GPU・TinyStories・fla はプロトコル＋Safety後。
A3完了まで README/論文に学習数値を書かない。

### Path A note — SDALM-10M 足場（枝 `sdalm-10m`）

枝 `sdalm-10m` に CPU・PyTorch の因果 LM 足場 `research/sdalm_10m/` がある。Core A1 で凍結した形状は **d_model=256, L=6, H=4, d_ff=1024, V=8000**。既定は **SDA-Recency**（学習は軟ゲート。硬ゲートは eval 切替のみ）。Expanded は `variant="expanded"` であり既定ではない。二つを混ぜた万能核にはしない。SDA 側で学習するのは β、τ、ゲート温度だけ（ほかは通常の埋め込み・射影・FFN）。archive ρ ∈ {small, base, large} は後続アブレーションであり、この足場では回さない。

パラメータ数は `python research/sdalm_10m/model.py` が印字する。`train_smoke.py` は固定 seed の CPU 数ステップで、有限勾配とスモーク損失の軌跡だけを見る。**学習済み層の有効性も LM 性能も主張しない。** hard 面の最良は全体最良ではない（hard = SDA-Recency、iid / old-skew = Expanded）。等メモリの万能核は未達。fla / Kimi Linear / KDA には依存しない。旧 README の不均等メモリ KDA 代理比較を勝ちとして引用しない。状態バイト数は dtype の item size で数え、`accepted_rate` は Recency 自身のスコアだけから出す（Expanded へ写さない。`hit_rate` は出さない）。設計は `research/sdalm_10m/DESIGN.md`。

## 未コミット成果物の扱い

`research/eqmem_hard*`・`hybrid_*`・`recency_adverse*`・`face_repro_ml2/`・本ファイル・`papers/SDA5_M-L1_limitations_JA.md` は**証跡としてリポジトリ内に残す**。公開 PR で main に載せる範囲は README の研究状況節・`RESEARCH_STATUS.md`・M-L1・face_repro を優先し、巨大 raw CSV は必要に応じて `.gitignore` または LFS 方針を別途決める（現時点では削除しない）。
