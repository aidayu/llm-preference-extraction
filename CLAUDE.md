# CLAUDE.md

対話から 3 軸嗜好（Liking / Wanting / Need）を構造化抽出し、知識グラフを構築する
リポジトリ。実験は Makefile 駆動（`extract → evaluate → plot`）。詳細は README を参照。

## Issue 駆動開発の規約

このリポジトリは Issue 駆動で開発する。フレームワークの全体設計は
`../docs/issue-driven-research-framework.md`（親ディレクトリ）を参照。

### 基本
- 実装作業は必ず Issue に紐づく。Issue のない実装を始めない。
- Issue 本文の「受け入れ条件」「結果の読み方」を正とみなす。
  そこから逸れそうになったら実装を進めず、本文の更新を先に提案すること。
- 判定基準を結果に合わせて後から書き換えてはならない。

### Git
- 着手時に `git switch -c <feat|exp>/<番号>-<slug>` でブランチを切る。
- コミットメッセージ末尾に `Refs #<番号>`。
- 完了時に PR を作成し、本文に `Closes #<番号>` を含める。**マージはユーザーが行う。**

### 実験の実行とコスト
- **実験を実行する前に必ず cost_estimator でコストを見積もり、金額を提示する**
  （`cd tools && uv run python -m cost_estimator --model <モデル>`）。見積もりなしに実行しない。
- 実行の担当は見積もり額で分ける:
  - **$0.50 以下**（動作確認・少数テスト、`--limit` 付き等）→ Claude が実行してよい。
    ただし実行前に見積もり額を提示すること。
  - **$0.50 超**（全データセット × 全モデルの抽出など）→ コマンドを提示し、実行はユーザー。
- ローカル ollama（gemma3:27b / llama3.1:8b）は API 課金なし。課金は OpenAI 系モデルでの抽出。
- このリポジトリに実験トラッカー（W&B / MLflow）は無い。結果はファイルで残る:
  - 抽出結果: `data/results/raw/experiments/<モデル>/`
  - 評価結果: `data/results/raw/evaluations/<モデル>/`
  - 集計: `data/results/summary/`、図表: `reports/figures/`
- Issue 固有の成果物（その Issue 専用の図・集計・レポート）を出す場合は、
  番号を接頭辞にしたサブディレクトリに置く。例: `reports/figures/12-cot-vs-baseline/`
- クローズ前に Issue へ「## 結論」コメントを書く。結果ファイルへのパスを必ず含める。

### スラッシュコマンド
- `/new-issue <やりたいこと>` — インタビュー形式で Issue を起票
- `/close-issue <番号>` — 事前の判定基準に照らして結論を起草しクローズ
