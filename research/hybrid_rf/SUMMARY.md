# SDA-Hybrid-RF（Arch M-D3d: Recency LRU pool + Routed multi-F）結果

実行: `OPENBLAS_NUM_THREADS=1 .venv/bin/python research/hybrid_rf/run_hybrid.py`（約15秒）。
実装: `hybrid.py`（更新 S[b]+=β k(v−S[b]ᵀk)ᵀ、b=argmax(k·P_r)；読出は Arch 疑似コードの残差補正 out=f+g(v_j−f_j)、切替版はアブレーション）、`data_iid.py`。eqmem_hard.py 非改変。
配分A: SDA予算と等KiB、R=round(48/97·R_exp)、残りをPへ（射影P_rもHybrid側で計上＝Expandedより不利な数え方）。
hard: R=48, P=490 → 387.75 KiB（Expanded 388.0）。i.i.d./old: B=128→R6,P70（51.75KiB）、B=256→R12,P130（99.75KiB）。
Hybrid β,τ は各suite/セルの CALIB seed（σ∈{0,0.4}）で同時格子選択。SDA-Recency β/τ=0.05/0.75 凍結。R スイープは未実施（配分A凍結）。

## σ=0 主判定: paired Δ=Expanded−Hybrid（正ならHybrid勝ち）
| suite | N,B | Expanded | Hybrid | SDA-Recency | Δ(Exp−Hyb) [95%CI] |
|---|---|---|---|---|---|
| hard (eqmem_hard) | 2048,1024 | 0.686 | 0.741 | **0.575** | −0.055 [−0.088,−0.022] |
| i.i.d.一様過去 | 512,128 | **0.594** | 0.654 | 0.680 | −0.060 [−0.077,−0.044] |
| | 512,256 | **0.415** | 0.463 | 0.445 | −0.048 [−0.061,−0.035] |
| | 2048,128 | **0.881** | 0.914 | 0.956 | −0.033 [−0.043,−0.023] |
| | 2048,256 | **0.812** | 0.852 | 0.893 | −0.040 [−0.052,−0.028] |
| v2 old-skew | 512,128 | **0.645** | 0.789 | 0.946 | −0.144 [−0.156,−0.132] |
| | 512,256 | **0.567** | 0.693 | 0.948 | −0.127 [−0.137,−0.117] |
| | 2048,128 | **0.872** | 0.927 | 0.991 | −0.055 [−0.063,−0.048] |
| | 2048,256 | **0.813** | 0.892 | 1.001 | −0.079 [−0.084,−0.074] |

σ=0.4（付記）: hard +0.242、i.i.d. 512系で +0.045/+0.146（2048系は差なし）、old-skew は全セル負。

## どの面で勝つか（正直に）
- Hybrid は **どの面でも最良にならない**「中間解」。hard では SDA-Recency（pool 1024本）に負け（rewrite_hit 0.546 vs 0.875）、i.i.d./old では Expanded に負ける。
- 対 SDA-Recency: i.i.d.（3/4セル）と old-skew（全セル）では有意に改善 ＝ 分割Fの B外補完は効いている。
- 対 Expanded: 正確poolはB外クエリに無価値で、R半減により圧縮側が弱くなる損の方が大きい。正確枠と圧縮枠のゼロサムが配分Aでは両負け。
- 残差補正 vs 切替: i.i.d./old では同値、hard では残差が良い（0.741 vs 0.799）。

## 判定: **No-go**（σ=0 で走らせた全9セルで Expanded−Hybrid の CI上限<0）
次の候補案: R/P スイープ（R→97に寄せるとExpanded、P→1024に寄せるとRecencyに縮退するだけの可能性が高い）、もしくはクエリ分布に応じた配分。test後の配分選択は holdout（20270001–）必須。LM主張なし。
注: summary_results の frac_inPool/mse_inPool は全手法とも Hybrid の pool 基準。
ファイル: hybrid.py, data_iid.py, run_hybrid.py, raw_results.csv, summary_results.csv, paired_results.csv, configuration.json, run.log
