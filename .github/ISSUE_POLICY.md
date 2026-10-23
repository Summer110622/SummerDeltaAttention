# Issue Policy / Issueポリシー

## Purpose / 目的
Issues track actionable bugs, feature proposals, documentation problems, and reproducible research questions for this educational, unreviewed research prototype.
Issueは、本教育用・未査読研究プロトタイプに関する、対応可能なバグ、機能提案、文書の問題、再現可能な研究上の疑問を管理します。

## Before opening / 投稿前
- Search existing open and closed issues and read the README and research limitations. Avoid duplicates.
- Do not disclose secrets, private datasets, personal information, or copyrighted materials without permission.
- Be respectful and constructive. Spam, harassment, and off-topic promotion may be closed.
- 既存のOpen/Closed Issue、README、研究上の制約を確認し、重複を避けてください。
- 秘密情報、非公開データ、個人情報、無許可の著作物を投稿しないでください。
- 建設的で礼節ある議論をお願いします。スパム、嫌がらせ、無関係な宣伝はクローズする場合があります。

## What to include / 記載事項
- Use the appropriate issue template and a specific title. English or Japanese is accepted.
- For bugs: expected/actual behavior, minimal reproduction, environment, traceback/logs (redacted), and test command.
- For features: motivation, proposed behavior, alternatives, compatibility and performance trade-offs.
- For research: hypothesis, experiment protocol, seeds, baselines, metrics, reproducibility steps, and known limitations.
- 適切なテンプレートと具体的なタイトルを使用してください。英語・日本語のいずれも受け付けます。
- バグ: 期待結果・実際の結果、最小再現手順、実行環境、機密情報を除いたログ、テストコマンド。
- 機能提案: 背景、期待する動作、代替案、互換性・性能上のトレードオフ。
- 研究: 仮説、実験手順、シード、比較手法、評価指標、再現方法、既知の制約。

## Scientific accuracy / 科学的な正確性
- Distinguish hypotheses from measured results. Synthetic exact-key recall is not a downstream language-model benchmark.
- The KDA-style proxy is not a trained Kimi Linear implementation. Do not claim general superiority, constant-memory exact recall, or linear-time complexity without evidence and accounting for archive costs.
- 仮説と測定結果を区別してください。合成的なexact-key recallは言語モデルの下流タスク評価ではありません。
- KDA風の代理モデルは学習済みKimi Linear実装ではありません。アーカイブのコストを含む根拠なしに、一般的な優位性、定数メモリでの完全想起、線形時間計算量を主張しないでください。

## Triage and closure / トリアージとクローズ
Maintainers may ask for clarification, label or prioritize issues, merge duplicates, or close issues that are out of scope, unreproducible after reasonable follow-up, inactive, or resolved. Opening an issue does not guarantee implementation or a response deadline. Substantial algorithm/protocol changes should be discussed in an issue before a PR; see [PR policy](../CONTRIBUTING.md) when available.
メンテナーは追加情報の依頼、ラベル付け、優先順位付け、重複の統合、対象外・合理的な確認後も再現不能・長期間更新なし・解決済みのIssueのクローズを行う場合があります。Issueの作成は実装や回答期限を保証しません。大規模なアルゴリズム・実験手順の変更はPR前にIssueで相談してください。PRについては[PRポリシー](../CONTRIBUTING.md)を参照してください（利用可能な場合）。
