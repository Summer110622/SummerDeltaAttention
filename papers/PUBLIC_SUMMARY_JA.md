# Summer Delta Attention — 公開用1ページ要約（2026-10-25）

**範囲:** 合成連想記憶・未学習 NumPy・等メモリ比較のみ。LM 性能は未検証。

## 分かったこと
- 等メモリでの最良核は**面に依存**する。
  - hard（上書き＋妨害）: **SDA-Recency** MSE 0.575（Recency KV 0.651、Expanded 0.686）。※hard の Recency 優位は当該生成（rewrite が後半）に依存し、old-skew では成立しない。
  - iid 一様過去: **Expanded DeltaNet**（512/128: 0.594 vs SDA-Recency 0.680）
  - old-skew: **Expanded DeltaNet**（512/128: 0.645 vs SDA-Recency 0.946）
- 3面同時に勝つ核を狙った H′・Hybrid-RF・Cond2 はすべて No-go。**等メモリ万能核は実証失敗**。仕様 H′ は **C=4 No-go**；C=16 は test 閲覧後の事後変更・holdout 未確認のため**採用しない・主結果にしない**。

## 言わないこと
- 「SDA が全面で勝つ」「万能」「KDA に勝った」「LM で有効」

## 再現
`bash research/face_repro_ml2/reproduce.sh`

詳細: `RESEARCH_STATUS.md` / `papers/SDA5_M-L1_limitations_JA.md`
