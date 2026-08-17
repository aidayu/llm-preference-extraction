"""抽出実験の API コスト見積もり。

run_extraction.py は「テスト対話1件につき chat completion を1回」というフラットな構造。
そのためトークン数を baseline 定数で近似せず、**実際に送るプロンプトを組み立てて実測**する。

入力の内訳（1コールあたり）:
    system  = few_shot_extract_template_cot.txt に few-shot 例を埋めたもの（全コール共通・固定）
    schema  = response_format の json_schema（全コール共通・固定）
    user    = "Extract preference from this dialogue.\n\n{dialogue_text}"（対話ごとに変動）

出力トークンは実測できないため、過去の実行結果（data/results/raw/experiments/）から
1件あたりの平均を取るか、--output-tokens で明示する。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import yaml

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[1]
PRICING_PATH = _HERE / "pricing.yaml"

sys.path.insert(0, str(PROJECT_ROOT))

from llm_preference_extraction.extractors import (  # noqa: E402
    create_few_shot_examples,
    load_prompt_template,
    load_schema,
)

# GPT-4o / GPT-5.x 系の encoding。tiktoken が未知のモデル名で落ちないよう固定で指定する。
DEFAULT_ENCODING = "o200k_base"

# テストおよび既定の見積もり対象データセット
DEFAULT_DATASET_FOR_TEST = PROJECT_ROOT / "data" / "ground_truth" / "test.json"

# 出力トークンの実測サンプルが無いときのフォールバック（過去 gemma4 実行の平均に基づく保守値）
FALLBACK_OUTPUT_TOKENS = 560


@dataclass
class Estimate:
    """1 モデル分の見積もり結果。"""

    model: str
    calls: int
    system_tokens: int
    schema_tokens: int
    user_tokens: int
    output_tokens_per_call: int
    input_cost: float
    output_cost: float
    is_local: bool
    cached_input_cost: float | None = None

    @property
    def input_tokens(self) -> int:
        return self.calls * (self.system_tokens + self.schema_tokens) + self.user_tokens

    @property
    def output_tokens(self) -> int:
        return self.calls * self.output_tokens_per_call

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

    @property
    def cached_total_cost(self) -> float | None:
        """プロンプトキャッシュが効いた場合の合計。対応外モデルは None。"""
        if self.cached_input_cost is None:
            return None
        return self.cached_input_cost + self.output_cost


def load_pricing(path: Path = PRICING_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _price(model: str, pricing: dict) -> dict:
    if model not in pricing["models"]:
        raise KeyError(
            f"未知のモデル: {model!r}。pricing.yaml の models: に定義してください"
        )
    return pricing["models"][model]


def _encoder(encoding_name: str = DEFAULT_ENCODING):
    import tiktoken

    return tiktoken.get_encoding(encoding_name)


def count_prompt_tokens(
    dataset_path: Path,
    few_shot_ids: list[int],
    encoding_name: str = DEFAULT_ENCODING,
) -> tuple[int, int, int, int]:
    """実際のプロンプトを組み立ててトークンを実測する。

    Returns:
        (テスト対話数, system トークン, schema トークン, user トークン合計)
    """
    enc = _encoder(encoding_name)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # run_extraction.py と同じ手順で system prompt を再現する
    few_shot_text = create_few_shot_examples(dataset, few_shot_ids)
    system_prompt = load_prompt_template().replace("{few_shot_examples}", few_shot_text)
    schema = load_schema()

    test_dialogues = [d for d in dataset if d["dialogue_id"] not in few_shot_ids]

    system_tokens = len(enc.encode(system_prompt))
    schema_tokens = len(enc.encode(json.dumps(schema)))
    user_tokens = sum(
        len(
            enc.encode(
                f"Extract preference from this dialogue.\n\n{d['original_dialogue']}"
            )
        )
        for d in test_dialogues
    )

    return len(test_dialogues), system_tokens, schema_tokens, user_tokens


def measure_output_tokens(
    results_glob: str = "data/results/raw/experiments/*/*.json",
    encoding_name: str = DEFAULT_ENCODING,
) -> int | None:
    """過去の実行結果から1コールあたりの平均出力トークンを実測する。

    抽出結果の JSON をそのまま出力とみなす。サンプルが無ければ None。
    """
    import glob

    enc = _encoder(encoding_name)
    counts: list[int] = []

    for path in sorted(glob.glob(str(PROJECT_ROOT / results_glob))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("results", []):
            payload = r.get("extracted_preferences")
            if payload:
                counts.append(len(enc.encode(json.dumps(payload, ensure_ascii=False))))

    if not counts:
        return None
    return round(sum(counts) / len(counts))


def estimate(
    *,
    model: str,
    dataset_path: Path,
    few_shot_ids: list[int],
    pricing: dict,
    output_tokens_per_call: int | None = None,
    runs: int = 1,
    encoding_name: str = DEFAULT_ENCODING,
) -> Estimate:
    """1 モデル分のコストを見積もる。

    runs は同一データセットを繰り返し流す回数（seed 違いの反復実験など）。
    """
    price = _price(model, pricing)

    n_dialogues, system_tokens, schema_tokens, user_tokens = count_prompt_tokens(
        dataset_path, few_shot_ids, encoding_name
    )

    if output_tokens_per_call is None:
        output_tokens_per_call = measure_output_tokens(encoding_name=encoding_name)
        if output_tokens_per_call is None:
            output_tokens_per_call = FALLBACK_OUTPUT_TOKENS

    calls = n_dialogues * runs
    total_user_tokens = user_tokens * runs

    est = Estimate(
        model=model,
        calls=calls,
        system_tokens=system_tokens,
        schema_tokens=schema_tokens,
        user_tokens=total_user_tokens,
        output_tokens_per_call=output_tokens_per_call,
        input_cost=0.0,
        output_cost=0.0,
        is_local=bool(price.get("is_local", False)),
    )

    if est.is_local:
        return est

    est.input_cost = est.input_tokens / 1_000_000 * price["input_per_mtok"]
    est.output_cost = est.output_tokens / 1_000_000 * price["output_per_mtok"]

    # プロンプトキャッシュ: system+schema は全コール共通なので、2回目以降はキャッシュ対象。
    # 1コール目のみ通常単価、残りは割引後単価で計算する。
    discount = price.get("cached_input_discount")
    if discount and calls > 1:
        fixed = est.system_tokens + est.schema_tokens
        rate = price["input_per_mtok"]
        uncached = (fixed + total_user_tokens) / 1_000_000 * rate
        cached = (fixed * (calls - 1)) / 1_000_000 * rate * (1 - discount)
        est.cached_input_cost = uncached + cached

    return est
