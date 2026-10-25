# M-L1 限界ドラフト（日本語・査読前）

**マイルストーン:** M-L1 · **日付:** 2026-10-25（Asia/Tokyo）  
**主張の上限:** 合成連想記憶・等メモリ比較に限る。LM性能は未検証。  
**禁止語:** 「有望な中間」「全面勝ち」「万能SOTA」「KDAに勝った」（主文では使わない）。

---

## 1. 面別 SOTA（σ=0・等メモリ帯・TEST 16 seed）— 先頭

| 面 | 最良 | MSE（代表） | 証跡 |
|---|---|---:|---|
| **hard**（rewrite+distractor, n=2048, ρ=0.5, B=1024） | **SDA-Recency** | **0.575 ± 0.020**（388 KiB）；次点 Recency KV **0.651**、Expanded **0.686** | `research/eqmem_hard_v2/` |
| **iid 一様過去** | **Expanded DeltaNet** | 例 512/128 → Expanded **0.594** / SDA-Recency **0.680**；512/256 → Expanded **0.415** | `research/hybrid_rf/` |
| **old-skew**（π=0） | **Expanded DeltaNet** | 例 512/128 → Expanded **0.645** / SDA-Recency **0.946** | `research/recency_adverse_v2/` |

hard: paired Δ（Expanded − SDA-Recency）= **+0.111 [+0.071, +0.151]**（その面の Go-1 通過）。SDA-Recency の Recency KV に対する上乗せ（0.651→0.575）は主に archive 外キーへの F 補完である。  
old-skew: 同符号が反転（例 512/128 で Expanded−SDA-Recency = **−0.301 [−0.313, −0.289]**）。  
→ **単一アーカイブ核は面依存**である。面をまたぐ単一の最良核はない。  
**注意:** hard 面の Go を主核凍結や全体最良と読むな（失敗核 H）。

## 2. 等メモリ万能核は実証失敗

等メモリ帯で hard・iid・old を同時に制する核は、以下の試行では**実証失敗**した。

| 失敗核 | 判定 | 要点 | 証跡 |
|---|---|---|---|
| **H（hard 過大主張）** | **主核不適格** | hard Go（SDA-Recency 0.575）は rewrite 最新保持＋生成が recency 有利な構造に依存。old-skew では Recency 全敗。**hard 勝ちを全体最良／主核凍結に外挿しない** | `research/eqmem_hard_v2/` · `research/recency_adverse_v2/` |
| **H′（SDA-Hyper-EqMem, 仕様 C=4）** | **No-go** | σ=0 MSE **0.750 ± 0.010**（386.5 KiB）。vs Expanded Δ=**−0.064 [−0.092, −0.036]**（CI上限<0） | `research/eqmem_hard_hyper/` |
| **Hybrid-RF**（配分A） | **No-go** | σ=0・全9セルで Expanded−Hybrid の CI上限<0。hard では Hybrid **0.741** vs Recency **0.575**。どの面でも最良でない（R/P ゼロサム） | `research/hybrid_rf/` |
| **Cond2**（`hybrid_switch`） | **No-go** | 書込統計で面は識別できるが性能未達。hard で誤 Exact（pool 内正解≈0.23 なのに hit≈0.55）、rewrite_hit **0.371≪0.875**、MSE≈**0.993**。σ=0・全9セルで Expanded 未達 | `research/hybrid_switch/` |

固定配分のゼロサム（正確プール枠 vs 圧縮状態枠）が残り、切替・ハッシュ索引でも解消しなかった。holdout（20270001–）での事後配分選択は未実施。[^c16]  
主判定は **σ=0**。σ=0.4 での部分逆転（例: Hybrid が hard で一時的に Expanded に勝つ余地）を万能化の根拠にしない。[^s04]

## 3. 限界リスト

1. **面依存:** hard の最良は Recency、iid / old-skew の最良は Expanded。核の順位は面で入れ替わる。
2. **等メモリ万能核は実証失敗:** H（過大主張）/ H′ / Hybrid / Cond2 はいずれも主核にならない。
3. **データ構造バイアス:** eqmem_hard_v2 の Recency 改善は archive 選定（最新値保持）が主因。一様過去など別生成では別途検証が必要。
4. **範囲:** 未学習 NumPy・合成タスクのみ。学習済み層・LM・GPU・ANN は未測定。早期の不均等メモリ KDA 代理比較は脚注へ格下げする。[^kda]
5. **主張禁止:** 「全面勝ち」「有望な中間」「万能SOTA」「LMで有効」「hard Go＝全体最良」は本証跡の外。

## 4. 証跡パス

- `research/eqmem_hard_v2/` — hard = Recency **0.575**（次点 Recency KV 0.651）
- `research/recency_adverse_v2/` — old-skew = Expanded；Recency 全敗
- `research/hybrid_rf/` — iid = Expanded；Hybrid **No-go**
- `research/hybrid_switch/` — Cond2 **No-go**
- `research/eqmem_hard_hyper/` — 案H′（C=4）**No-go**

[^kda]: **脚注（旧比較・格下げ）:** 早期の不均等メモリ合成ベンチでは、校正 SDA（100 KiB）が未学習 KDA 代理（4 KiB）より低い MSE（0.4721 vs 0.9336）を示した。メモリ約25倍差のため、主主張の「勝ち」には使わない。一次ソース: `research/summary_results.csv`（n=512, noise=0, retention=0.5）。

[^c16]: 案H′の C=16 事後アブレーションは Expanded に有意勝ちうるが、test 閲覧後の変更であり holdout 未確認のため、本限界節では仕様 C=4 の **No-go** のみを採用する。

[^s04]: σ=0.4 は付録指標。主判定・万能核判定は σ=0 に固定する。

---

**一文:** 面別SOTAは hard=SDA-Recency 0.575（次点 Recency KV 0.651）、iid・old=Expanded。H / H′ / Hybrid / Cond2 は主核不適格。**等メモリ万能核は実証失敗**である。
