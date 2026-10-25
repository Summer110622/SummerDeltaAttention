# M-D3e SDA-Cond2（書込統計→2モード切替）— 判定: **No-go**

実装: `run_cond2.py`（eqmem_hard.py / hybrid_rf / adverse 系は非改変、helper を import のみ）。OPENBLAS_NUM_THREADS=1、TEST 16 seeds、HOLDOUT **未実施**。

## 構成（configuration.json）
- hard: R=64, P=309 → 387.9 KiB（Expanded R=97 388 KiB）。iid/old: 各セルの Expanded 予算に R=round(64/97·R_exp)、残りを P（hybrid_rf と同じスケーリング規則）。
- CALIB_SEEDS のみで**全面・全セル・σ{0,0.4}をプールした1構成**を選択（面ラベル不使用、Arch格子フル 162点）。
- 凍結: β=0.2, θ_rw=0.05, τ_R=0.85, τ_E=0.85, W=128（W・θ は同値タイ）。
- ρ_rw: hard≈0.357（→ Mode-R 100%）、iid/old=0（全key distinct → Mode-E 100%）。信号は面を完全分離するが、それでも負け。

## σ=0 主結果（MSE, TEST平均）
| 面 | Expanded | SDA-Recency | **Cond2** | Δ Expanded−Cond2 [95%CI] |
|---|---|---|---|---|
| hard n2048 | 0.686 | **0.575** | 0.993 | −0.307 [−0.338, −0.276] |
| iid 512/128 | **0.594** | 0.680 | 0.674 | −0.080 [−0.095, −0.065] |
| iid 512/256 | **0.415** | 0.445 | 0.570 | −0.156 [−0.167, −0.145] |
| iid 2048/128 | **0.881** | 0.956 | 0.893 | −0.012 [−0.022, −0.002] |
| iid 2048/256 | **0.812** | 0.893 | 0.822 | −0.010 [−0.018, −0.002] |
| old 512/128 | **0.645** | 0.946 | 0.753 | −0.108 [−0.120, −0.095] |
| old 512/256 | **0.567** | 0.949 | 0.721 | −0.155 [−0.166, −0.143] |
| old 2048/128 | **0.872** | 0.991 | 0.909 | −0.037 [−0.046, −0.029] |
| old 2048/256 | **0.813** | 1.001 | 0.851 | −0.038 [−0.044, −0.032] |

全9セルで CI上限<0 → **Go条件（3面 CI下限>0）不成立**。old/iid では SDA-Recency には勝つ（old 全セル有意）が Expanded 未達。

## 診断（hard σ=0）
- Mode-R 使用率 1.0、pool hit率 0.548、うち正しいkeyが pool 内 0.234 → **誤ヒット多数**（15–30°distractor が cos≥0.85 を越え、残差なし切替で誤 v をそのまま返す）。
- 非hit側の分割F（R=64, β=0.2）が out-of-pool MSE 1.30 と弱い。共通β格子{.2,.4,.7}で全面1値に縛られ、hard 単体最適（Expanded β=0.4）とずれた。
- rewrite_hit 0.371（Recency 0.875、Expanded 0）。
- σ=0.4: hard は Expanded と差なし（+0.007 [−0.019,+0.033]）、iid 512/256 のみ Cond2 有意勝ち、old は全敗。

## 結論
切替信号自体は書込統計で面を完全に識別できたが、(1) P=309 の小プールでは hard の正解 key が 23% しか残らず、近接distractorでの誤切替が支配、(2) R を減らした分割F は Expanded 未達。固定配分のゼロサムが切替でも解消しない。holdout は未実施（No-goのため不要と判断）。

ファイル: raw_results.csv / summary_results.csv / paired_results.csv / calib_grid.csv / configuration.json / run.log
