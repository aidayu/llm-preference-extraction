# cost_estimator — 抽出実験の API コスト見積もり

`experiments/run_extraction.py` を回す前に、モデル別の API コストを見積もる。
**見積もり自体は API を呼ばず無料**。

## 評価側ツールとの違い

`llm-preference-evaluation/tools/cost_estimator` は `personas × methods × seeds` の
3コンポーネント構成（generation / preprocess / judge）を前提にしており、抽出実験には
そのまま使えない。抽出実験は「テスト対話1件 → chat completion 1回」のフラットな構造。

代わりに本ツールは **baseline 定数を使わず、実際に送るプロンプトを組み立ててトークンを実測する**。
`few_shot_extract_template_cot.txt` + few-shot 例 + `schema_template_cot.json` +
各対話の user メッセージを `tiktoken`（`o200k_base`）で数えるので、入力トークンは概算ではなく厳密値。

## 使い方

```bash
cd tools
uv run python -m cost_estimator --model gpt-5.6-terra
```

50件版データセット・複数モデル比較:

```bash
uv run python -m cost_estimator \
    --model gpt-5.6-terra,gpt-4o,gpt-4o-mini --full
```

出力（例）:

```
データセット: dailydialog_annotated_integrated.json
Few-shot ids: [0, 1, 2]（テスト対象から除外）
テスト対話数: 47 × runs=1 = 47 コール
1コールあたり入力: system 1,691 + schema 568 (固定) + user 平均 128
1コールあたり出力: 545 tok（過去実行結果からの実測平均）

model             calls      in_tok    out_tok    cost($)
----------------------------------------------------------
gpt-5.6-terra        47     112,214     25,615     0.6648
gpt-4o               47     112,214     25,615     0.5367
gpt-4o-mini          47     112,214     25,615     0.0322

プロンプトキャッシュ有効時（system+schema が2コール目以降キャッシュヒット）:
  gpt-5.6-terra      0.4310  (-0.2338, -35%)
```

### 主なオプション

| オプション | 既定 | 意味 |
|---|---|---|
| `--model` | `gpt-5.6-terra` | モデル名のカンマ区切り（`pricing.yaml` 定義済みのもの） |
| `--full` | off | 50件版データセットを使う（既定は `test.json` の11件） |
| `--dataset` | - | データセットを明示指定 |
| `--few-shot-ids` | `0,1,2` | Few-shot に使う id。テスト対象から除外される |
| `--runs` | 1 | 同一データセットを流す回数（反復実験用） |
| `--output-tokens` | 実測 | 1コールあたり出力トークンを明示 |

## トークンの数え方

**入力**は3要素に分けて実測する。`--few-shot-ids` や対話の中身を変えると自動で反映される。

- `system` … few-shot 例を埋め込んだプロンプト全体（**全コール共通・固定**）
- `schema` … `response_format` の json_schema（**全コール共通・固定**）
- `user` … `"Extract preference from this dialogue.\n\n{dialogue}"`（対話ごとに変動）

固定分（system + schema = 2,259 tok）が入力の **95% 以上**を占め、対話本体（平均128 tok）は
ごく一部。これがプロンプトキャッシュの効きが大きい理由。

**出力**は事前実測できないため、`data/results/raw/experiments/` の過去実行結果から
抽出JSONのトークン数を平均する（現状 545 tok/件）。サンプルが無ければ
`FALLBACK_OUTPUT_TOKENS` を使う。`--output-tokens` で上書き可能。

## 前提と注意

- **出力トークンは gemma4 実行結果からの流用**。GPT系はJSON構造が同じでも
  `chain_of_thought` の長さが変わりうるので、GPT で1回流したら実測し直すのが望ましい。
- **プロンプトキャッシュ行は上限値**（system+schema が毎回ヒットした理想ケース）。
  OpenAI のキャッシュは最小トークン数・TTL の条件があるため、実額はこの間に収まる。
- **メッセージのラッパ分は未計上**。role 等のオーバーヘッド数トークン/コールは無視している。

## ファイル

| ファイル | 役割 |
|---|---|
| `pricing.yaml` | モデル別単価・is_local・キャッシュ割引率・出典 |
| `estimator.py` | プロンプト再構成＋トークン実測＋コスト計算 |
| `__main__.py` | CLI |
| `test_estimator.py` | 単価・回数スケール・ローカル無料・キャッシュ割引の回帰テスト |
