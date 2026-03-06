"""嗜好抽出実験スクリプト

Few-shotプロンプトを用いてデータセットから嗜好を抽出する。

使用方法:
    python experiments/run_extraction.py

元ファイル: preference_kg/experiments/run_extraction.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_preference_extraction.extractors import (
    create_client,
    create_few_shot_examples,
    extract_preferences,
    load_prompt_template,
    load_schema,
)

load_dotenv()

# =====================================================================
# === ユーザー設定 ===
# =====================================================================
# データセットパス
# DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "ground_truth" / "dailydialog_annotated_integrated.json"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "ground_truth" / "test.json"

# デフォルトモデル設定
DEFAULT_MODEL_NAME = "gpt-4o-mini"

# Few-shot用のdialogue_id
# FEW_SHOT_IDS = [0, 18, 46]
FEW_SHOT_IDS = [0, 1, 2]
# =====================================================================


def load_dataset(dataset_path: Path) -> list:
    """データセットを読み込む"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_experiment(
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    base_url: str | None = None,
    api_key: str | None = None,
    save_path: Path | None = None,
) -> list:
    """
    抽出実験を実行する

    Args:
        model_name: 使用するモデル名
        dataset_path: データセットのパス
        base_url: Ollama等のローカルサーバーURL（Noneの場合はOpenAI API）
        api_key: APIキー
        save_path: 結果を保存するパス

    Returns:
        list: 抽出結果
    """
    print(f"\n{'='*60}")
    print(f"=== 嗜好抽出実験開始: {model_name} ===")
    print(f"{'='*60}")
    print(f"データセット: {dataset_path}")
    print(f"Few-shot dialogue_ids: {FEW_SHOT_IDS}")

    # クライアント作成
    client = create_client(base_url, api_key)
    print(f"Client: {'Ollama' if base_url else 'OpenAI API'}")

    # データセット読み込み
    print("\n[1/4] データセット読み込み中...")
    dataset = load_dataset(dataset_path)
    print(f"総対話数: {len(dataset)}")

    # Few-shot例の作成
    print("\n[2/4] Few-shot例作成中...")
    few_shot_examples_text = create_few_shot_examples(dataset, FEW_SHOT_IDS)

    # プロンプトテンプレート読み込み
    print("\n[3/4] プロンプトテンプレート読み込み中...")
    prompt_template = load_prompt_template()
    system_prompt = prompt_template.replace("{few_shot_examples}", few_shot_examples_text)

    # スキーマ読み込み
    schema = load_schema()

    # テスト対象の対話を取得（Few-shot例を除く）
    test_dialogues = [d for d in dataset if d["dialogue_id"] not in FEW_SHOT_IDS]
    print(f"テスト対話数: {len(test_dialogues)}")

    # 嗜好抽出実行
    print("\n[4/4] 嗜好抽出実行中...")
    results = []

    for dialogue_data in tqdm(test_dialogues, desc=f"Processing [{model_name}]"):
        dialogue_id = dialogue_data["dialogue_id"]
        dialogue_text = dialogue_data["original_dialogue"]

        # 抽出実行
        extraction_result = extract_preferences(
            client,
            model_name,
            dialogue_text,
            dialogue_id,
            system_prompt,
            schema,
        )

        # 元のアノテーションも保持
        result_with_annotation = {
            "dialogue_id": dialogue_id,
            "original_dialogue": dialogue_text,
            "translated_dialogue": dialogue_data.get("translated_dialogue", ""),
            "ground_truth_annotations": dialogue_data.get("annotations", []),
            "extracted_preferences": extraction_result,
        }

        results.append(result_with_annotation)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_metadata = {
        "experiment_info": {
            "timestamp": timestamp,
            "dataset_path": str(dataset_path),
            "few_shot_ids": FEW_SHOT_IDS,
            "total_test_dialogues": len(test_dialogues),
            "model": model_name,
        },
        "results": results,
    }

    if save_path:
        # 結果保存
        print("\n結果保存中...")

        if save_path is None:
            output_dir = PROJECT_ROOT / "data" / "results" / "raw" / "experiments" / model_name
        else:output_dir = save_path / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"experiment_results_{timestamp}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(experiment_metadata, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 実験完了！")
        print(f"結果保存先: {output_path}")
        print(f"処理対話数: {len(results)}")

        # 簡易統計
        total_extracted = sum(
            len(r["extracted_preferences"].get("preferences", [])) for r in results
        )
        total_ground_truth = sum(len(r["ground_truth_annotations"]) for r in results)
        print(f"\n統計:")
        print(f"  - 抽出された嗜好数: {total_extracted}")
        print(f"  - Ground truth嗜好数: {total_ground_truth}")

    return experiment_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="嗜好抽出実験スクリプト")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)

    args = parser.parse_args()

    model = args.model
    dataset_path = args.dataset

    # Ollamaモデルのパターンにマッチするかチェック
    ollama_patterns = ["llama", "gemma", "mistral", "phi", "qwen", "codellama"]
    is_ollama = any(pattern in model.lower() for pattern in ollama_patterns)

    if is_ollama:
        run_experiment(
            model_name=model,
            dataset_path=dataset_path,
            base_url="http://localhost:11434/v1",
            api_key=model,
        )
    else:
        run_experiment(
            model_name=model,
            dataset_path=dataset_path,
            base_url=None,
            api_key=None,
        )
