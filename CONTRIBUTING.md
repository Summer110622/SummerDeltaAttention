# Pull Request Policy / プルリクエストポリシー

This repository is an educational, unreviewed research prototype. Contributions are welcome, but scientific claims must remain reproducible and carefully qualified.

本リポジトリは教育目的の未査読研究プロトタイプです。貢献を歓迎しますが、科学的な主張には再現性と適切な限定条件が必要です。

## Scope / 対象

- Open an issue first for substantial algorithm changes, new dependencies, or changes to published experiment protocols. Small fixes and documentation improvements may go directly to a PR.
- Keep each PR focused; avoid unrelated refactoring.
- Never include secrets, private datasets, or third-party content without suitable rights and attribution.

- アルゴリズムの大幅な変更、依存関係の追加、公開済み実験手順の変更は、原則として先にIssueで相談してください。小規模な修正や文書改善は直接PRで構いません。
- PRは一つの目的に絞り、無関係なリファクタリングを含めないでください。
- 秘密情報、非公開データ、適切な権利・出典のない第三者コンテンツを含めないでください。

## Submission / 提出要件

1. Branch from `main` and open a PR targeting `main`; use a draft PR for unfinished work.
2. Describe the motivation, exact changes, possible limitations, linked issues, and how to verify the changes. English or Japanese is acceptable; a bilingual summary is appreciated.
3. Run `python -m unittest discover -s tests -v` for code changes. If relevant, run `python -m examples.quickstart`. Report commands and results, or explain why they could not run.
4. Add or update tests for behavior changes and update documentation when public APIs, algorithms, or experimental claims change.
5. For benchmark or research changes, report seeds, dataset/synthetic setup, baselines, metrics, hardware/software versions when relevant, and a reproduction command. Distinguish measured results from hypotheses; disclose memory and computational costs and fairness limitations.
6. Do not describe the KDA-style proxy as a trained Kimi Linear implementation, or synthetic recall measurements as downstream language-model results. Avoid unsupported superiority, linear-time, and exact-recall claims.

1. `main` からブランチを作成し、`main` 宛てにPRを作成してください。作業途中の場合はDraft PRを使用してください。
2. 目的、変更内容、制約、関連Issue、検証方法を記載してください。英語・日本語のいずれかで構いませんが、日英併記の要約を歓迎します。
3. コード変更では `python -m unittest discover -s tests -v` を実行し、関連する場合は `python -m examples.quickstart` も実行してください。コマンドと結果、または実行できなかった理由を記載してください。
4. 動作変更にはテストを追加・更新し、公開API・アルゴリズム・実験上の主張の変更時には文書も更新してください。
5. ベンチマーク・研究変更では、乱数シード、データ／合成実験条件、比較手法、指標、必要に応じた実行環境、再現コマンドを明記してください。測定結果と仮説を区別し、メモリ・計算量・比較の公平性に関する制約を開示してください。
6. KDA風の代理モデルを学習済みKimi Linear実装と表現したり、合成記憶実験を言語モデルの下流評価と表現したりしないでください。根拠のない優位性・線形時間・完全想起の主張は避けてください。

## Review and merge / レビューとマージ

- A maintainer reviews changes for correctness, scope, tests, reproducibility, and documentation. Address review comments before merging.
- Prefer squash merging after approval and successful applicable checks. Maintainers may request changes or decline proposals that are out of scope.
- This document describes the review policy; mandatory approvals, status checks, and branch protection must be configured separately in GitHub repository settings.

- メンテナーは正確性、変更範囲、テスト、再現性、文書を確認します。マージ前にレビュー指摘に対応してください。
- 承認と該当チェックの成功後、原則としてSquash Mergeを推奨します。対象外の提案には修正やクローズをお願いする場合があります。
- 本文はレビュー方針です。必須承認・必須チェック・ブランチ保護の強制設定はGitHubのリポジトリ設定で別途行う必要があります。
