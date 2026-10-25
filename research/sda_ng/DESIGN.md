# SDA-NG design freeze

Status: frozen for this branch. Synthetic associative recall only. No LM claim. No pull request.

Working title: **SDA-NG**. This is not a universal core and not a new face winner.

## English (short)

Keep the hard-face winner skeleton: no-decay delta state `F` (β = 0.05) and a Recency-B archive (LRU on distinct key ids, latest value). Replace only the hard accept.

```
cos_j = max_i ⟨q, k_i⟩
T     = max(temperature, ε)          # ε = 1e-6, so T > 0
g     = σ((cos_j − τ) / T)           # σ(x) = 1 / (1 + exp(−x)),  g ∈ (0, 1)
out   = Fᵀq + g · (v_j − Fᵀk_j)
```

Frozen constants, not fit on TEST: `τ = 0.75`, `temperature = 0.10`.

Optional CALIB-only threshold, also implemented and not a TEST result:

```
τ_cal ← Quantile_{p=0.3}({cos_j on CALIB queries})
```

For a hard gate that quantile targets an accept rate near `1 − p = 0.7` when cosines are continuous. NumPy's linear quantile is the definition used in code.

案2 (margin / piecewise-linear gate) is **not implemented**. It is named below only as a future hard-face option.

`research/eqmem_hard.py` is not on `best-face-results`. The smoke stream is a documented reconstruction (see データ). It is not a bit-exact replay of the published TEST MSE 0.575.

## 日本語（凍結本文）

### 名前

SDA-NG（working title）。hard 面の勝者骨格からゲートだけを変える作業名。全面勝者の宣言ではない。

### 残すもの

- 忘却なし delta 則の `F`。更新は `F ← F + β k (v − Fᵀk)ᵀ`。β = 0.05（SDA-Recency の committed 値。本ブランチでは再探索しない）。
- アーカイブは Recency-B。相異なる key id の LRU。容量 B = round(ρ n) = 1024（n = 2048, ρ = 0.5）。同一 id の再書き込みはスロットを増やさず、値を最新にし、その id を最新側へ動かす。
- 読み出しは残差補正の形のまま。最近傍 `(k_j, v_j)` の選び方（アーカイブ上のコサイン最大）は変えない。

`best-face-results` に `eqmem_hard.py` / `run_v2.py` は無い。参照実装の `recency_positions` は、文書どおり「保持された各 id の最新書き込み位置」であり、ライブラリ本体の Algorithm R（reservoir）ではない。ライブラリの公開 API は変えていない。

### 変えるもの（案1。本ブランチの唯一のゲート）

ハード受理 `g = 1[cos_j ≥ τ]` を、連続なソフト残差重みに置き換える。

```
cos_j = max_i ⟨q, k_i⟩
T     = max(temperature, ε)          # ε = 1e-6。T の clamp_min
g     = σ((cos_j − τ) / T)           # σ(x) = 1 / (1 + e^{−x})
out   = Fᵀq + g · (v_j − Fᵀk_j)
```

凍結定数:

| 記号 | 値 | 扱い |
| --- | --- | --- |
| τ | 0.75 | committed hard threshold。TEST で再較正しない |
| temperature | 0.10 | 本ブランチの凍結温度。TEST スイープではない |
| ε | 1e-6 | `T ← max(temperature, ε)`。0 以下を渡してもステップゲートに戻さない |
| β | 0.05 | F のステップ。ゲートとは独立 |

temperature = 0.10 の理由（事前の失敗モードへの合わせであり、TEST 最適化ではない）: 以前の LM 経路では exact-key（π = 1）でも max cosine ≈ 0.574 が凍結 τ ≈ 0.75 を下回り、`accepted_rate = 0` のまま ρ の CE が識別できなかった。この点で

`g = σ((0.574 − 0.75) / 0.10) = σ(−1.76) ≈ 0.147 > 0.05`

となり、ハードゲートが閉じ切る余弦でも重みが 0 に張り付かない。一方 `cos = 1` では `g = σ(2.5) ≈ 0.924 < 1` なので、アーカイブに正解がある exact-key でも出力は `v` に厳密一致しない。補正が `(1 − g)` だけ `Fᵀk` 側に残る。これは式の帰結であり、スモークの MSE でそのまま出す。

空アーカイブのときは最近傍が無いので `g = 0`、`out = Fᵀq` とする（シグモイドを無理に掛けない）。

### CALIB のみの任意 τ（実装する。主結果は凍結 τ）

```
τ_cal ← Quantile_{p=0.3}({cos_j : CALIB クエリの top-1 コサイン})
```

定義は `numpy.quantile` の既定（linear）。ハードゲート `1[cos ≥ τ_cal]` として読むと、コサインが連続なら受理率はおよそ `1 − p = 0.7` になる。ソフトゲートの `g` は二値の受理率ではない。τ_cal は CALIB コサインから決める推論時定数で、TEST seed には当てない。

スモークは CALIB 8 seed のコサインをプールして τ_cal を計算し、**同じ CALIB クエリに戻して**診断欄に書く。これは in-sample であり、ホールドアウト性能ではない。

合成 exact-key ではアーカイブ内クエリの cos が 1 に点在する。`p = 0.3` がその点質量に落ちると `τ_cal = 1` になり、ハード受理率 0.7 は実現しない。その場合も観測値をそのまま書き、目標 0.7 に届いたとは書かない。このオプションは、コサインが τ 未満に連続して乗る局面（上記 LM 経路のような分布）向けの閾値移動であり、hard 面の exact 復元を良くする機構だとは主張しない。

主スモークの成否は凍結 τ = 0.75 の `mean_g` で判定する。

### 案2（未実装。将来の hard 面オプション）

Margin gate（区分線形）は本ブランチのコードに存在しない。将来、hard 面で `cos = 1` の厳密復元を保ったまま閾値未満に幅を持たせたい場合の候補として、式だけ残す。

```
g = clip((cos − τ_lo) / (τ_hi − τ_lo), 0, 1)
```

τ_hi 以上で `g = 1`、τ_lo 以下で `g = 0` になる。シグモイドと違い exact-key の縮みをゼロにできる可能性がある。採用も実装もしない。Hybrid-RF / Cond2 の再導入でもない。

### データ（スモーク用の再構成）

公開プロトコルから取った定数:

- n = 2048、U = ⌊0.85 n⌋ = 1740、D = 32、V = 16、ρ = 0.5、B = 1024、Q = 128
- クエリは σ = 0 の exact key（`q = k`）。目標値は当該 id の最新 value
- CALIB seed は `20250001`–`20250008`。TEST `20261023`–`20261038` は走らせない

ストリームの形（再構成。公開 TEST の hit 率にフィットさせていない）:

- 前半 1024: early-stable。1 回だけ書く
- 後半 1024: distractor 408 を 1 回 + rewrite key 308 を 2 回。2 回目側の出現が新しい value
- クエリは rewrite 64 + once-only（early と distractor の混合）64

この分割では後半の相異なる id は 716 ≤ 1024 なので、Recency-B は rewrite と distractor をすべて残し、early-stable は直近 308 件だけ残す。公開 TEST の rewrite_hit = 0.875、stable_hit = 0.487、SDA-Recency MSE = 0.575 ± 0.020 とは一致しない。一致を主張しない。ハードゲートとの MSE 差は、この再構成・同一 seed の中だけで読む。

### 非目標

- 全 face で勝つ単一核。iid と old-skew の等メモリ勝者は Expanded DeltaNet のまま。
- LM 性能、学習済み層、GPU、速度、ρ の CE が識別できた、という主張。
- Hybrid-RF、Cond2、H′ の復活。該当ディレクトリは変更しない。
- ライブラリ `summer_delta_attention` の既定ゲートを変えること。
- Pull request を開くこと。
- TEST 16 seed。スモークが `mean_g > 0` でも、本ラウンドの完了条件は CALIB ログまで。

### 成功条件（このブランチ）

1. 本メモで案1の式と凍結定数が固定されている。
2. `research/sda_ng/reference.py` がそのソフトゲートと Recency-B を実装している。
3. CALIB exact-key で凍結 τ の `mean_g > 0`。望ましくは `frac(g > 0.05) > 0`。`mean_g ≤ 0` なら TEST に進まず、ログに失敗を書いて止める。
4. 同じ seed でハードゲート `g = 1[cos ≥ 0.75]` の MSE を併記する。ソフトが悪ければ悪いと書く。`MSE < 1e-3` の rewrite_hit は、`g(cos=1) < 1` のためハードより落ちてよい。主指標は MSE。

### 面の勝者（変更しない）

| 面 | 勝者 | 証跡 |
| --- | --- | --- |
| hard（rewrite + distractor） | SDA-Recency（本スケルトンの元） | `research/eqmem_hard_v2/` |
| iid（一様過去） | Expanded DeltaNet (eq-mem) | `research/hybrid_rf/` |
| old-skew（adverse_v2） | Expanded DeltaNet (eq-mem) | `research/recency_adverse_v2/` |

Hybrid-RF と Cond2 は No-go のまま。参照もしない方向へ拡張しない。
