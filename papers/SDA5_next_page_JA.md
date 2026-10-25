# SDA-5：次の1ページ（査読前草稿・合成ベンチ限定）

**著者：summer** · 2026-10-25 · 区分：未査読の研究メモ  
**根拠データ：** `research/summary_results.csv`（開発seed 8本で校正、未使用16 seedで測定）  
**注意：** 以下の数値は合成連想記憶（exact-key / noisy-key recall）に限る。言語モデルの次トークン予測・下流タスク性能は**未検証**。

---

## 貢献

1. **方式の明示的分解：** 忘却なしDelta Ruleの高速状態 `F`、容量制限付きKVアーカイブ、類似度しきい値による二値ゲート残差補正を一つの参照実装にまとめた（SDA-5）。
2. **合成ベンチでの校正効果：** N=512・ノイズなし・保存率50%で、開発データから選んだ `β=0.05, τ=0.75` の校正SDAは MSE **0.4721 ± 0.0099**（SEM）、状態+KV **100 KiB**。未校正の元設定は同条件で **0.9422 ± 0.0221**。
3. **等メモリ以降の面依存（主主張の土台）：** 等メモリ帯では hard の最良は SDA-Recency（MSE **0.575**）、iid・old-skew の最良は Expanded。単一アーカイブ核の横断最良はない。同アーカイブ Indexed KV との差など、不均等メモリ上の旧比較は脚注に格下げする。[^kda]

## 手法（要約）

KeyをL2正規化し、状態を

\[
e_t=v_t-F_{t-1}^{\mathsf T}k_t,\qquad
F_t=F_{t-1}+\beta\,k_te_t^{\mathsf T}
\]

で更新する（乗法的忘却なし）。アーカイブから Query に最も近い Key を全件走査し、類似度が `τ` 以上のときだけ残差補正する。

\[
o(q)=F^{\mathsf T}q+g(q)\big(v_{i^*}-F^{\mathsf T}k_{i^*}\big),\quad
g(q)=\mathbf1[q^{\mathsf T}k_{i^*}\ge\tau].
\]

実験プロトコル：Key次元32、Value次元16、Query128、系列長 `N∈{128,512,2048}`、Queryノイズ `σ∈{0,0.4,1}`、保存率 `{0.1,0.25,0.5,1.0}`。比較実験は全書き込み後の固定乱数KV選抜であり、ストリーム中のreservoirサンプル（`examples/quickstart.py`）とはプロトコルが異なる。

## 限界

**面別 SOTA（σ=0・等メモリ）— 先頭**

| 面 | 最良 | 代表 MSE | 証跡 |
|---|---|---:|---|
| hard | **SDA-Recency** | **0.575 ± 0.020** | `eqmem_hard_v2` |
| iid | **Expanded** | 例 0.594（512/128） | `hybrid_rf` |
| old-skew | **Expanded** | 例 0.645（Recency 0.946） | `recency_adverse_v2` |

- **面依存：** 単一アーカイブ核の順位は面で入れ替わる。「全面勝ち」は言えない。
- **等メモリ万能核は実証失敗：** 案**H**（Hyper-EqMem C=4）/ **Hybrid-RF** / **Cond2** はいずれも **No-go**（`eqmem_hard_hyper`, `hybrid_rf`, `hybrid_switch`）。
- **タスク範囲：** 合成 recall のみ。LM学習・次トークン予測は未検証。
- **メモリ–精度：** 誤差低下の多くは KV 明示保存による。固定サイズ無損失連想記憶ではない。
- **計算量：** `O(d_k d_v+B(d_k+d_v))` メモリ、Queryごと `O(B d_k)`。GPU・ANN 未測定。
- **禁止：** 「有望な中間」「全面勝ち」「万能SOTA」。詳細は `papers/SDA5_M-L1_limitations_JA.md`。

[^kda]: 脚注（旧・不均等メモリ）: 校正SDA 100 KiB vs 未学習KDA代理 4 KiB で MSE 0.4721 vs 0.9336。メモリ約25倍差のため主文の勝ち主張には使わない（`summary_results.csv`）。

---

**再現：** `OPENBLAS_NUM_THREADS=1 python research/experiment.py` → `research/figures.py`  
**出典数値：** papers/SDA_vs_KDA_JA.md §3–5 および `research/summary_results.csv`（n=512, noise=0, retention=0.5）。
