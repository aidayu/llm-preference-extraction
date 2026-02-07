"""暗黙的嗜好推論ロジック

対話データから明示されていない暗黙的嗜好を推論する。

元ファイル:
- preference_kg/experiments2/batch/extract_preferences.py (infer_implicit_preferences)
- preference_kg/experiments2/prompts/inference.py
"""

import json
from pathlib import Path
from typing import Optional

from openai import OpenAI


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "data" / "prompts"


def load_implicit_inference_prompt(filename: str = "implicit_inference_prompt.txt") -> str:
    """暗黙的嗜好推論用プロンプトを読み込む"""
    filepath = PROMPTS_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def infer_implicit_preferences(
    client: OpenAI,
    model_name: str,
    dialogue_text: str,
    dialogue_id: int,
    explicit_preferences: list[dict],
) -> dict:
    """
    暗黙的嗜好を推論する

    Args:
        client: OpenAI client instance
        model_name: 使用するモデル名
        dialogue_text: 対話テキスト
        dialogue_id: 対話ID
        explicit_preferences: 既に抽出済みの明示的嗜好リスト

    Returns:
        暗黙的嗜好の推論結果
    """
    explicit_entities = [p.get("entity", "") for p in explicit_preferences]
    explicit_info = f"\n既に抽出済みの明示的嗜好: {explicit_entities}" if explicit_entities else ""

    try:
        prompt = load_implicit_inference_prompt()
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"以下の対話から暗黙的嗜好を推論してください。{explicit_info}\n\n対話:\n{dialogue_text}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        result = json.loads(completion.choices[0].message.content)
        return result

    except Exception as e:
        print(f"Error in implicit inference for dialogue {dialogue_id}: {e}")
        return {
            "implicit_preferences": [],
            "error": str(e),
        }


def integrate_preferences(
    dialogue_id: int,
    user_id: str,
    explicit_result: dict,
    implicit_result: dict,
    generation_model: str,
    analysis_model: str,
) -> dict:
    """
    明示的嗜好と暗黙的嗜好を統合

    Args:
        dialogue_id: 対話ID
        user_id: ユーザーID
        explicit_result: 明示的嗜好抽出結果
        implicit_result: 暗黙的嗜好推論結果
        generation_model: 対話生成に使用したモデル
        analysis_model: 抽出・推論に使用したモデル

    Returns:
        統合された嗜好データ
    """
    explicit_preferences = []
    for pref in explicit_result.get("preferences", []):
        pref_copy = pref.copy()
        explicit_preferences.append(pref_copy)

    implicit_preferences = implicit_result.get("implicit_preferences", [])

    return {
        "dialogue_id": dialogue_id,
        "user_id": user_id,
        "generation_model": generation_model,
        "analysis_model": analysis_model,
        "chain_of_thought": explicit_result.get("chain_of_thought", ""),
        "explicit_preferences": explicit_preferences,
        "implicit_preferences": implicit_preferences,
    }
