# A3 評価プロトコル（正本）

このファイルが A3 の正本である。英語ファイル [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) はポインタだけで、閾値・シード・手順をここに重複して書かない。

対象は凍結形状の **SDA-Recency のみ**（d_model=256, L=6, H=4, d_ff=1024, V=8000, base archive_cap=32）。CPU スモークまでの手順である。学習済み層が言語で効く、という主張はしない。GPU、TinyStories、fla は、このプロトコルと Safety の後まで開かない。

## ゲート

学習は軟ゲートだけとする。

```text
g̃ = σ((cos − τ) / T)
```

評価は硬ゲート `1[cos ≥ τ]` とする。`hard_gate_eval` は `eval()` のときだけ有効で、学習ステップでは無視する。硬ゲートの straight-through estimator（STE）は A3 の外である。

β の初期値は 0.05、τ の初期値は 0.75。A2/A3 のスモークで AdamW が更新してよいのは `beta_logit`、`tau_logit`、`gate_log_temp` だけである。埋め込み、QKV、FFN、出力射影は勾配を診断用に持てるが、オプティマイザのパラメータ群には入れない。

## ベースライン

同じ形状で、次の二つだけを対照にする。

1. **softmax 全注意** — 可能ならパラメータ数を揃え、状態メモリも揃えられる範囲で揃える。実装に fla は要らない。
2. **無減衰 Delta** — アーカイブなし、ゲートなし。Expanded（幅を広げた状態）ではない。

Expanded と Hybrid はこのプロトコルのベースラインに入れない。面別の合成結論（hard = SDA-Recency、iid / old-skew = Expanded、万能核は失敗）は書き換えない。

## タスク

主セルは **MQAR hard**（次トークンの連想想起を主タスクにした学習側のセル）である。iid と old-skew は参照セルとして残し、そこから面の勝者を選び直さない。

## 指標

必須の記録は **next-token cross-entropy** だけである。MSE と accept / hit は内部ログに限る。README、論文、`RESEARCH_STATUS.md` にその数値を写さない。スモークスクリプトが標準出力に出す損失軌跡も、評価結果として転記しない。

`hit_rate` と `accepted_rate` は、その手法自身のマスクから計算する。校正済み SDA の率を他手法の行へコピーしない。メモリのバイト数は `dtype.itemsize` で数える。`* 8` で float64 を仮定しない。

## シード

帯は重ねない。

| 帯 | 範囲 | 本数 | 用途 |
|---|---|---:|---|
| CALIB | 20250001–20250008 | 8 | τ / β / T の選択、早期停止、f16 ゲートの判定 |
| TEST | 20261023–20261038 | 16 | 最終報告の一度だけ。それまで封印 |
| HOLDOUT | 20270001–20270008 | 8 | 事後の確認用。選択には使わない |
| LM / MQAR | 20300001– | 開区間 | 学習バッチ。上の三帯と混ぜない |

τ、β、温度 T、および早期停止は CALIB だけで決める。TEST を見てから閾値や停止を変えない。TEST は最終報告で一度開く。

20261023 は TEST 帯の先頭であり、f16 ゲートの任意診断（`--smoke-test-seed`）に使ってよい。診断は閾値を変えない。判定シードには入れない。

## f16 dual-write ゲート（ロック）

スクリプト: `research/sdalm_10m/gate_f16_dualwrite.py`。

入力は CALIB シード 20250001–20250008 からその場で作る合成 KV だけである。LM/MQAR 帯 20300001– の学習バッチをゲートに使うのは漏洩であり、無効（exit 2）とする。

float64 参照と float16 dual-write（高速状態 F とアーカイブ K, V の両方をその dtype で書く）を、同じ入力で比べる。

| 対象 | 規則 |
|---|---|
| accept マスク、hit マスク | 完全一致。率は手法ごと。他手法から写さない |
| 連続量 o, F, K, V | `atol = rtol = 1e-3` |
| 上書き判定（f16） | `cosine >= 1 - 8 * finfo(float16).eps`（約 8e-3）。f64 参照の上書き許容は 1e-9 |
| 閾値の置き場 | スクリプト先頭の定数。CLI や smoke シードから代入しない |
| 終了 | exit 0 = PASS。それ以外は FAIL。最後に 1 行の JSON |

ρ の 3 点（[`ABLATION_RHO.md`](ABLATION_RHO.md)）は、このゲートが PASS するまで走らせない。PASS のあと、探索の 3 点は CALIB シードだけを使う `ablation_rho.py` が記録する。TEST と HOLDOUT は開かない。LM 帯 20300001– は校正に混ぜない。数値は `ABLATION_RHO.md` と `logs/ablation_rho_calib.json` に置き、このファイルには写さない。

## 計算の範囲

いま走らせる CPU の範囲は、スモーク、上のゲート、およびゲート PASS 後の CALIB ρ グリッドである。GPU、TinyStories、fla、SOTA の宣言は、プロトコルと Safety が済むまで行わない。DeltaNet、Gated DeltaNet、Kimi Linear / KDA は引用先であり、このプロトコルの測定対象ではない。
