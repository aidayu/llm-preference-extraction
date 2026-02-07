# llm-preference-extraction

LLMを用いた対話からの嗜好抽出と知識グラフ構築のためのツールキット。

## 概要

このプロジェクトは、対話データからユーザーの嗜好を抽出し、知識グラフとして構造化するためのツールを提供します。

### 主な機能

- **嗜好抽出 (実験1)**: Few-shot プロンプトによる明示的嗜好の抽出と評価
- **対話アプリ (実験2)**: Streamlit ベースの嗜好把握対話システム
- **知識グラフ構築**: 抽出された嗜好からKGトリプレットを生成

## セットアップ

```bash
# リポジトリをクローン
git clone <repository-url>
cd llm-preference-extraction

# 環境構築 (uv使用)
make setup

# 仮想環境をアクティベート
source .venv/bin/activate

# .envファイルを作成
cp .env.example .env
# OPENAI_API_KEY を設定
```

## ディレクトリ構造

```
llm-preference-extraction/
├── README.md              # このファイル
├── Makefile               # セットアップ・実行コマンド
├── pyproject.toml         # 依存関係定義
│
├── llm_preference_extraction/ # コアモジュール
│   ├── extractors.py          # 嗜好抽出ロジック
│   ├── reasoners.py           # 暗黙的推論ロジック
│   ├── graph_builder.py       # 知識グラフ構築
│   └── evaluation/            # 評価モジュール
│
├── experiments/           # 実験1: 評価スクリプト
│   ├── run_extraction.py  # 抽出実行
│   ├── run_evaluation.py  # 評価実行
│   └── plot_evaluation.py # 結果可視化
│
├── app/                   # 実験2: Streamlitアプリ
│   ├── main.py            # メインUI
│   └── components/        # UI部品
│
├── data/
│   ├── prompts/           # プロンプトテンプレート
│   ├── ground_truth/      # 正解データ
│   ├── results/           # 評価結果
│   └── output_samples/    # 生成KGサンプル
│
└── notebooks/             # 分析・可視化ノートブック
```

## 使い方

### 実験1: 抽出精度評価

```bash
# 抽出実行
make extract

# 評価実行
make evaluate

# 結果グラフ描画
make plot
```

### 実験2: 対話アプリ

```bash
make app
# ブラウザで http://localhost:8501 を開く
```

## ライセンス

MIT License
