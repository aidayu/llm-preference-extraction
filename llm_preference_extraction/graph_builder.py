"""知識グラフ構築ロジック

抽出された嗜好をKGトリプレットに変換し、JSONとして保存する。

元ファイル:
- preference_kg/experiments2/batch/build_kg.py
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def build_triplets_for_user(preferences_data: list[dict], user_id: str) -> list[dict]:
    """
    嗜好データを知識グラフトリプレットに変換

    Returns triplets in two formats:
    - Explicit: head=user_id, relation=combined_axis, tail=entity
    - Implicit: head=user_id, relation="implicit_preference", tail=sentence
    """
    triplets = []

    for pref_record in preferences_data:
        source_timestamp = pref_record.get("source_timestamp", "")
        dialogue_id = pref_record.get("dialogue_id", 0)

        # Process explicit preferences (structured)
        for pref in pref_record.get("explicit_preferences", []):
            if not pref.get("entity") or not pref.get("combined_axis"):
                continue

            triple = {
                "head": user_id,
                "relation": pref["combined_axis"],
                "tail": pref["entity"],
                "type": "explicit",
                "attributes": {
                    "polarity": pref.get("polarity", "neutral"),
                    "intensity": pref.get("intensity", "medium"),
                    "context_tags": pref.get("context_tags", []),
                    "original_mention": pref.get("original_mention", ""),
                    "dialogue_id": dialogue_id,
                    "source_timestamp": source_timestamp,
                },
            }
            triplets.append(triple)

        # Process implicit preferences (dict with inference and original_mention)
        for pref in pref_record.get("implicit_preferences", []):
            # Handle both old format (string) and new format (dict)
            if isinstance(pref, str):
                sentence = pref
                original_mention = ""
            elif isinstance(pref, dict):
                sentence = pref.get("inference", "")
                original_mention = pref.get("original_mention", "")
            else:
                continue

            if not sentence:
                continue

            triple = {
                "head": user_id,
                "relation": "implicit_preference",
                "tail": sentence,
                "type": "implicit",
                "attributes": {
                    "original_mention": original_mention,
                    "dialogue_id": dialogue_id,
                    "source_timestamp": source_timestamp,
                },
            }
            triplets.append(triple)

    return triplets


def build_knowledge_graph(
    preferences_data: list[dict],
    user_id: str,
    generation_model: str = "unknown",
    analysis_model: str = "unknown",
) -> dict:
    """
    嗜好データから知識グラフを構築

    Args:
        preferences_data: 嗜好抽出結果のリスト
        user_id: ユーザーID
        generation_model: 対話生成に使用したモデル
        analysis_model: 抽出・推論に使用したモデル

    Returns:
        知識グラフデータ（metadata + triples）
    """
    triplets = build_triplets_for_user(preferences_data, user_id)

    # Count types and relations
    type_counts = {"explicit": 0, "implicit": 0}
    relation_counts = {}
    for triple in triplets:
        triple_type = triple.get("type", "explicit")
        type_counts[triple_type] = type_counts.get(triple_type, 0) + 1
        rel = triple["relation"]
        relation_counts[rel] = relation_counts.get(rel, 0) + 1

    output_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "user_id": user_id,
            "generation_model": generation_model,
            "analysis_model": analysis_model,
            "total_triples": len(triplets),
            "explicit_count": type_counts.get("explicit", 0),
            "implicit_count": type_counts.get("implicit", 0),
        },
        "triples": triplets,
    }

    return output_data


def save_knowledge_graph(kg_data: dict, output_path: Path) -> None:
    """
    知識グラフをJSONファイルに保存

    Args:
        kg_data: build_knowledge_graph の出力
        output_path: 出力ファイルパス
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kg_data, f, indent=2, ensure_ascii=False)


def load_preferences_from_jsonl(filepath: Path) -> list[dict]:
    """
    JSONLファイルから嗜好データを読み込む

    Args:
        filepath: 嗜好データのJSONLファイルパス

    Returns:
        嗜好レコードのリスト
    """
    if not filepath.exists():
        return []

    preferences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                preferences.append(json.loads(line))
    return preferences
