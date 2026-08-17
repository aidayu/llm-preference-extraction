"""嗜好抽出実験スクリプト

Few-shotプロンプトを用いてデータセットから嗜好を抽出する。

使用方法:
    python experiments/run_extraction.py

元ファイル: preference_kg/experiments/run_extraction.py
"""

import argparse
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
# FEW_SHOT_IDS = [0, 18, 46] は 50件版にのみ存在するため、こちらを既定にする。
# 動作確認用の test.json を使うときは --dataset と --few-shot-ids を併せて指定すること。
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "ground_truth" / "dailydialog_annotated_integrated.json"
# DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "ground_truth" / "test.json"

# デフォルトモデル設定
DEFAULT_MODEL_NAME = "gpt-4o-mini"

# 実験結果の保存先
DEFAULT_SAVE_DIR = PROJECT_ROOT / "data" / "results" / "raw" / "experiments"

# Few-shot用のdialogue_id
# 他モデル（gpt-5.2 / gpt-4o / gpt-4o-mini / gemma3 / llama3.1）の実験と揃える正式設定。
# これらの id は 50件版データセットにのみ存在する（test.json は id 0-10 のみ）。
FEW_SHOT_IDS = [0, 18, 46]
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
    save_dir: Path | None = DEFAULT_SAVE_DIR,
    few_shot_ids: list[int] | None = None,
) -> list:
    """
    抽出実験を実行する

    Args:
        model_name: 使用するモデル名
        dataset_path: データセットのパス
        base_url: Ollama等のローカルサーバーURL（Noneの場合はOpenAI API）
        api_key: APIキー
        save_dir: 結果を保存するディレクトリ
        few_shot_ids: Few-shot に使う dialogue_id（Noneなら FEW_SHOT_IDS）

    Returns:
        list: 抽出結果
    """
    few_shot_ids = list(FEW_SHOT_IDS if few_shot_ids is None else few_shot_ids)

    print(f"\n{'='*60}")
    print(f"=== 嗜好抽出実験開始: {model_name} ===")
    print(f"{'='*60}")
    print(f"データセット: {dataset_path}")
    print(f"Few-shot dialogue_ids: {few_shot_ids}")

    # クライアント作成
    client = create_client(base_url, api_key)
    print(f"Client: {'Ollama' if base_url else 'OpenAI API'}")

    # データセット読み込み
    print("\n[1/4] データセット読み込み中...")
    dataset = load_dataset(dataset_path)
    print(f"総対話数: {len(dataset)}")

    # Few-shot例の作成
    print("\n[2/4] Few-shot例作成中...")
    # 指定 id がデータセットに無いと警告だけ出て例が減り、条件が変わったまま完走してしまう。
    # 実験条件の取り違えは結果を無言で壊すのでここで止める。
    available_ids = {d["dialogue_id"] for d in dataset}
    missing_ids = [i for i in few_shot_ids if i not in available_ids]
    if missing_ids:
        raise SystemExit(
            f"Few-shot dialogue_id {missing_ids} が {dataset_path.name} に存在しません。\n"
            f"  データセットの id 範囲: {min(available_ids)}-{max(available_ids)}\n"
            f"  --dataset か --few-shot-ids を見直してください。"
        )
    few_shot_examples_text = create_few_shot_examples(dataset, few_shot_ids)

    # プロンプトテンプレート読み込み
    print("\n[3/4] プロンプトテンプレート読み込み中...")
    prompt_template = load_prompt_template()
    system_prompt = prompt_template.replace("{few_shot_examples}", few_shot_examples_text)

    # スキーマ読み込み
    schema = load_schema()

    # テスト対象の対話を取得（Few-shot例を除く）
    test_dialogues = [d for d in dataset if d["dialogue_id"] not in few_shot_ids]
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
            "few_shot_ids": few_shot_ids,
            "total_test_dialogues": len(test_dialogues),
            "model": model_name,
        },
        "results": results,
    }

    if save_dir is not None:
        # 結果保存
        print("\n結果保存中...")

        output_dir = save_dir / model_name
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
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help="実験結果の保存先ディレクトリ",
    )
    parser.add_argument(
        "--few-shot-ids",
        type=str,
        default=None,
        help=f"Few-shot に使う dialogue_id のカンマ区切り（既定: {','.join(map(str, FEW_SHOT_IDS))}）",
    )

    args = parser.parse_args()

    model = args.model
    dataset_path = args.dataset
    save_dir = args.save_dir
    few_shot_ids = (
        [int(x) for x in args.few_shot_ids.split(",") if x.strip()]
        if args.few_shot_ids
        else None
    )

    # Ollamaモデルのパターンにマッチするかチェック
    ollama_patterns = ["llama", "gemma", "mistral", "phi", "qwen", "codellama"]
    is_ollama = any(pattern in model.lower() for pattern in ollama_patterns)

    if is_ollama:
        run_experiment(
            model_name=model,
            dataset_path=dataset_path,
            base_url="http://localhost:11434/v1",
            api_key=model,
            save_dir=save_dir,
            few_shot_ids=few_shot_ids,
        )
    else:
        run_experiment(
            model_name=model,
            dataset_path=dataset_path,
            base_url=None,
            api_key=None,
            save_dir=save_dir,
            few_shot_ids=few_shot_ids,
        )
