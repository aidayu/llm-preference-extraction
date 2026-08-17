"""嗜好抽出ロジック

対話データからFew-shotプロンプトを用いて明示的嗜好を抽出する。

元ファイル:
- preference_kg/experiments/run_extraction.py
- preference_kg/experiments2/batch/extract_preferences.py
"""

import json
import os
from importlib.resources import files
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Paths
# パッケージ同梱のプロンプト（インストール先からでも解決できるようリソース参照する）
PROMPTS_DIR = files("llm_preference_extraction") / "prompts"


def load_prompt_template(template_name: str = "few_shot_extract_template_cot.txt") -> str:
    """Few-shotプロンプトテンプレートを読み込む"""
    return (PROMPTS_DIR / template_name).read_text(encoding="utf-8").strip()


def load_schema(schema_name: str = "schema_template_cot.json") -> dict:
    """JSONスキーマを読み込む"""
    return json.loads((PROMPTS_DIR / schema_name).read_text(encoding="utf-8"))


def load_few_shot_examples(filename: str = "few_shot_examples.json") -> list[dict]:
    """パッケージ同梱のFew-shot例（DailyDialog dialogue_id 0/18/46）を読み込む"""
    return json.loads((PROMPTS_DIR / filename).read_text(encoding="utf-8"))


def build_extraction_prompt(
    dataset: Optional[list[dict]] = None,
    few_shot_ids: Optional[list[int]] = None,
    template_name: str = "few_shot_extract_template_cot.txt",
) -> str:
    """
    抽出用のsystem promptを組み立てる

    テンプレート中の {few_shot_examples} をFew-shot例で置換する。
    dataset未指定の場合はパッケージ同梱のFew-shot例を使う。

    Args:
        dataset: Few-shot例の供給元データセット（Noneなら同梱例）
        few_shot_ids: Few-shot例に使うdialogue_id（Noneならdataset全件）

    Returns:
        str: {few_shot_examples} を置換済みのsystem prompt
    """
    if dataset is None:
        dataset = load_few_shot_examples()
    if few_shot_ids is None:
        few_shot_ids = [d["dialogue_id"] for d in dataset]

    examples_text = create_few_shot_examples(dataset, few_shot_ids)
    return load_prompt_template(template_name).replace("{few_shot_examples}", examples_text)


# temperature の指定を受け付けないモデル名の接頭辞。
# GPT-5 系（gpt-5.6-terra 等）は既定値(1)のみ対応で、0 を渡すと 400 エラーになる。
NO_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def supports_temperature(model_name: str) -> bool:
    """モデルが temperature の明示指定を受け付けるか。

    再現性のため既定では temperature=0 を使いたいが、GPT-5 系・o シリーズは
    既定値しか受け付けない。該当モデルでは呼び出し側でパラメータを省略する。
    """
    normalized = model_name.lower().lstrip("/")
    return not normalized.startswith(NO_TEMPERATURE_PREFIXES)


def create_client(base_url: Optional[str] = None, api_key: Optional[str] = None) -> OpenAI:
    """
    モデル設定に応じたOpenAI clientを作成

    Args:
        base_url: Ollama等のローカルサーバーURL（Noneの場合はOpenAI API）
        api_key: APIキー（Ollamaの場合はモデル名、OpenAIの場合は環境変数から）

    Returns:
        OpenAI: クライアントインスタンス
    """
    if base_url:
        return OpenAI(base_url=base_url, api_key=api_key or "ollama")
    else:
        return OpenAI()


def create_few_shot_examples(dataset: list[dict], few_shot_ids: list[int]) -> str:
    """
    データセットからFew-shot例を作成する

    Args:
        dataset: 全データセット（dialogue_id, original_dialogue, annotationsを含む）
        few_shot_ids: Few-shot用のdialogue_idリスト

    Returns:
        Few-shot例のテキスト
    """
    examples_text = ""

    for idx, dialogue_id in enumerate(few_shot_ids, 1):
        dialogue_data = next((d for d in dataset if d["dialogue_id"] == dialogue_id), None)

        if dialogue_data is None:
            print(f"Warning: dialogue_id {dialogue_id} not found in dataset")
            continue

        dialogue_text = dialogue_data["original_dialogue"]

        preferences = []
        for ann in dialogue_data.get("annotations", []):
            axis = ann.get("axis", "")
            sub_axis = ann.get("sub_axis", "")
            combined_axis = f"{axis}__{sub_axis}"

            preference = {
                "reasoning": f"Entity '{ann.get('entity', '')}' is mentioned with {ann.get('polarity', '')} sentiment",
                "combined_axis": combined_axis,
                "entity": ann.get("entity", ""),
                "original_mention": ann.get("entity", ""),
                "context_tags": ann.get("context", []) if ann.get("context") != ["None"] else [],
                "polarity": ann.get("polarity", "neutral"),
                "intensity": ann.get("intensity", "medium"),
            }
            preferences.append(preference)

        extraction = {
            "dialogue_id": dialogue_id,
            "user_id": "user",
            "preferences": preferences,
        }

        examples_text += f"\n### Example {idx}:\n"
        examples_text += f"**Dialogue:**\n{dialogue_text}\n\n"
        examples_text += f"**Extraction:**\n{json.dumps(extraction, indent=2, ensure_ascii=False)}\n"

    return examples_text


def extract_preferences(
    client: OpenAI,
    model_name: str,
    dialogue_text: str,
    dialogue_id: int,
    system_prompt: str,
    schema: dict,
) -> dict:
    """
    Few-shotプロンプトを使用して対話から嗜好を抽出する

    Args:
        client: OpenAI client instance
        model_name: 使用するモデル名
        dialogue_text: 対話テキスト
        dialogue_id: 対話ID
        system_prompt: システムプロンプト
        schema: JSONスキーマ

    Returns:
        抽出結果（JSON）
    """
    # GPT-5 系は temperature を既定値(1)しか受け付けず、0 を渡すと 400 になる。
    # その場合はパラメータ自体を省略する（既定値が使われる）。
    kwargs = {}
    if supports_temperature(model_name):
        kwargs["temperature"] = 0

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Extract preference from this dialogue.\n\n{dialogue_text}",
                },
            ],
            response_format={"type": "json_schema", "json_schema": schema},
            **kwargs,
        )

        result = json.loads(completion.choices[0].message.content)
        return result

    except Exception as e:
        print(f"Error extracting preferences for dialogue_id {dialogue_id}: {e}")
        return {
            "chain_of_thought": "",
            "dialogue_id": dialogue_id,
            "user_id": "user",
            "preferences": [],
            "error": str(e),
        }


def format_dialogue_for_extraction(dialogue_history: list[dict]) -> str:
    """Format dialogue history as text for LLM input."""
    lines = []
    for msg in dialogue_history:
        role = "User" if msg["role"] == "user" else "System"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)
