"""嗜好抽出ロジック

対話データからFew-shotプロンプトを用いて明示的嗜好を抽出する。

元ファイル:
- preference_kg/experiments/run_extraction.py
- preference_kg/experiments2/batch/extract_preferences.py
"""

import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = DATA_DIR / "prompts"


def load_prompt_template(template_name: str = "few_shot_extract_template_cot.txt") -> str:
    """Few-shotプロンプトテンプレートを読み込む"""
    filepath = PROMPTS_DIR / template_name
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_schema(schema_name: str = "schema_template_cot.json") -> dict:
    """JSONスキーマを読み込む"""
    filepath = PROMPTS_DIR / schema_name
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


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
            temperature=0,
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
