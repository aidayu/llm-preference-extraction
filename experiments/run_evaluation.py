"""嗜好抽出結果の評価スクリプト

src.evaluation モジュールを使用して抽出結果を評価する。
- 対話ごとにOptimal Matchingで評価
- Micro/Macro/Weighted F1で集計

使用方法:
    python experiments/run_evaluation.py

元ファイル: preference_kg/experiments/run_evaluation.py
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_preference_extraction.evaluation import (
    AggregatedAccuracy,
    AggregatedMetrics,
    DialogueResult,
    aggregate_all_metrics,
    evaluate_dialogue,
    split_combined_axis,
)

# =====================================================================
# === ユーザー設定 ===
# 既定値はNone。評価対象はCLI引数で指定してください。
# =====================================================================
EXPERIMENT_RESULTS_PATH = None
# =====================================================================

RESULTS_ROOT = PROJECT_ROOT / "data" / "results"


def load_experiment_results(filepath: Path) -> dict:
    """実験結果ファイルを読み込む"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_predictions(extracted_prefs: dict) -> list[dict]:
    """抽出結果を評価用の形式に変換する"""
    if "error" in extracted_prefs:
        return []

    predictions_raw = extracted_prefs.get("preferences", [])
    predictions = []

    for pred in predictions_raw:
        axis, sub_axis = split_combined_axis(pred.get("combined_axis", ""))
        pred_converted = pred.copy()
        pred_converted["axis"] = axis
        pred_converted["sub_axis"] = sub_axis
        predictions.append(pred_converted)

    return predictions


def evaluate_experiment(
    experiment_data: dict,
) -> tuple[list[DialogueResult], dict[str, AggregatedMetrics], AggregatedAccuracy] | None:
    """
    実験結果を評価する

    Args:
        experiment_data: 実験データ全体

    Returns:
        (dialogue_results, aggregated_metrics, accuracy): 対話ごとの結果と集計結果
    """
    results = experiment_data.get("results", [])

    if len(results) == 0:
        print("評価対象データがありません。")
        return None

    print(f"\n--- 評価開始: {len(results)}件の対話 ---")

    dialogue_results = []

    for result_entry in results:
        dialogue_id = result_entry["dialogue_id"]
        ground_truths = result_entry.get("ground_truth_annotations", [])
        extracted_prefs = result_entry.get("extracted_preferences", {})
        predictions = convert_predictions(extracted_prefs)

        # 対話ごとの評価
        result = evaluate_dialogue(
            dialogue_id=str(dialogue_id),
            ground_truths=ground_truths,
            predictions=predictions,
        )
        dialogue_results.append(result)

        # 対話ごとのサマリー表示
        status = (
            "SKIP"
            if result.n_gt == 0
            else ("PERFECT" if result.perfect_tp == result.n_gt else "PARTIAL")
        )
        print(
            f"[ID:{dialogue_id}] GT={result.n_gt}, Pred={result.n_pred}, "
            f"Entity-F1={result.entity_f1:.2%}, Axis-F1={result.axis_f1:.2%} -> {status}"
        )

    # 集計
    aggregated, accuracy = aggregate_all_metrics(dialogue_results)

    return dialogue_results, aggregated, accuracy


def print_aggregated_summary(
    aggregated: dict[str, AggregatedMetrics], accuracy: AggregatedAccuracy
):
    """集計結果を表示する"""
    print("\n" + "=" * 80)
    print("評価結果サマリー (Micro / Macro / Weighted F1)")
    print("=" * 80)

    header = f"{'Metric':<20} {'Micro-F1':>10} {'Macro-F1':>10} {'Weighted-F1':>12}"
    print(header)
    print("-" * 60)

    for name, metrics in aggregated.items():
        row = f"{name:<20} {metrics.micro_f1:>10.2%} {metrics.macro_f1:>10.2%} {metrics.weighted_f1:>12.2%}"
        print(row)

    print("=" * 80)

    # 条件付き分類精度を表示
    print("\n" + "=" * 80)
    print(f"条件付き分類精度 (マッチしたペア内でのAccuracy, N={accuracy.total_matched})")
    print("=" * 80)

    header = f"{'Attribute':<20} {'Micro-Acc':>12} {'Macro-Acc':>12}"
    print(header)
    print("-" * 50)

    acc_data = [
        ("Axis", accuracy.axis_micro_accuracy, accuracy.axis_accuracy),
        ("Sub-Axis", accuracy.sub_axis_micro_accuracy, accuracy.sub_axis_accuracy),
        ("H-Axis", accuracy.h_axis_micro_accuracy, accuracy.h_axis_accuracy),
        ("Polarity", accuracy.polarity_micro_accuracy, accuracy.polarity_accuracy),
        ("Intensity", accuracy.intensity_micro_accuracy, accuracy.intensity_accuracy),
        ("Context", accuracy.context_micro_accuracy, accuracy.context_accuracy),
        ("Perfect", accuracy.perfect_micro_accuracy, accuracy.perfect_accuracy),
    ]

    for attr, micro, macro in acc_data:
        print(f"{attr:<20} {micro:>12.2%} {macro:>12.2%}")

    print("=" * 80)


def save_aggregated_results(
    aggregated: dict[str, AggregatedMetrics],
    accuracy: AggregatedAccuracy,
    dialogue_results: list[DialogueResult],
    experiment_info: dict,
    output_path: Path,
):
    """集計結果をCSVに保存する"""
    total_gt = sum(r.n_gt for r in dialogue_results)
    total_pred = sum(r.n_pred for r in dialogue_results)
    total_matched = sum(r.n_matched for r in dialogue_results)

    with open(output_path, "w", encoding="utf-8") as f:
        # 実験情報
        f.write("Experiment Information,,,,\n")
        f.write(f"Timestamp,{experiment_info.get('timestamp', '')},,,\n")
        f.write(f"Model,{experiment_info.get('model', '')},,,\n")
        f.write(f"Few-shot IDs,\"{experiment_info.get('few_shot_ids', '')}\",,,\n")
        f.write(
            f"Total Test Dialogues,{experiment_info.get('total_test_dialogues', len(dialogue_results))},,,\n"
        )
        f.write(f"Total GT,{total_gt},,,\n")
        f.write(f"Total Pred,{total_pred},,,\n")
        f.write(f"Total Matched Pairs,{total_matched},,,\n")
        f.write(",,,,\n")

        # 集計結果
        f.write("Metric,Micro-F1,Macro-F1,Weighted-F1,Total TP\n")
        for name, metrics in aggregated.items():
            f.write(
                f"{name},{metrics.micro_f1:.4f},{metrics.macro_f1:.4f},{metrics.weighted_f1:.4f},{metrics.total_tp}\n"
            )

        f.write(",,,,\n")

        # 条件付き分類精度
        f.write("Conditional Classification Accuracy (within matched pairs),,,,\n")
        f.write("Attribute,Micro-Accuracy,Macro-Accuracy,,\n")
        f.write(f"Axis,{accuracy.axis_micro_accuracy:.4f},{accuracy.axis_accuracy:.4f},,\n")
        f.write(
            f"Sub-Axis,{accuracy.sub_axis_micro_accuracy:.4f},{accuracy.sub_axis_accuracy:.4f},,\n"
        )
        f.write(
            f"H-Axis,{accuracy.h_axis_micro_accuracy:.4f},{accuracy.h_axis_accuracy:.4f},,\n"
        )
        f.write(
            f"Polarity,{accuracy.polarity_micro_accuracy:.4f},{accuracy.polarity_accuracy:.4f},,\n"
        )
        f.write(
            f"Intensity,{accuracy.intensity_micro_accuracy:.4f},{accuracy.intensity_accuracy:.4f},,\n"
        )
        f.write(
            f"Context,{accuracy.context_micro_accuracy:.4f},{accuracy.context_accuracy:.4f},,\n"
        )
        f.write(
            f"Perfect,{accuracy.perfect_micro_accuracy:.4f},{accuracy.perfect_accuracy:.4f},,\n"
        )

        f.write(",,,,\n")

        # 詳細（Precision/Recallも含む）
        f.write("Detailed Metrics,,,,\n")
        f.write("Metric,Type,Precision,Recall,F1\n")
        for name, metrics in aggregated.items():
            f.write(
                f"{name},Micro,{metrics.micro_precision:.4f},{metrics.micro_recall:.4f},{metrics.micro_f1:.4f}\n"
            )
            f.write(
                f"{name},Macro,{metrics.macro_precision:.4f},{metrics.macro_recall:.4f},{metrics.macro_f1:.4f}\n"
            )
            f.write(
                f"{name},Weighted,{metrics.weighted_precision:.4f},{metrics.weighted_recall:.4f},{metrics.weighted_f1:.4f}\n"
            )

    print(f"結果を保存しました: {output_path}")


def main(
    experiment_results_path: Path | None = EXPERIMENT_RESULTS_PATH,
    evaluation_output_dir: Path | None = None,
):
    """メイン評価関数"""
    if experiment_results_path is None:
        print("評価対象の結果ファイルが指定されていません。")
        print("例: python experiments/run_evaluation.py --results <path-to-results.json>")
        sys.exit(2)

    print("=== 実験結果評価開始 ===")
    print(f"実験結果ファイル: {experiment_results_path}")

    # 出力先の設定
    if evaluation_output_dir is None:
        # パスからモデル名とタイムスタンプを抽出
        _exp_path = Path(experiment_results_path)
        _filename = _exp_path.stem
        _timestamp_match = re.search(r"(\d{8}_\d{6})", _filename)
        experiment_timestamp = (
            _timestamp_match.group(1)
            if _timestamp_match
            else datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        model_name = _exp_path.parent.name
        evaluation_output_dir = RESULTS_ROOT / "raw" / "evaluations" / model_name / experiment_timestamp

    print(f"評価結果出力先: {evaluation_output_dir}")

    print("\n[1/4] 実験結果読み込み中...")
    experiment_data = load_experiment_results(experiment_results_path)
    experiment_info = experiment_data.get("experiment_info", {})

    print(f"実験タイムスタンプ: {experiment_info.get('timestamp')}")
    print(f"モデル: {experiment_info.get('model')}")
    print(f"Few-shot IDs: {experiment_info.get('few_shot_ids')}")
    print(f"テスト対話数: {experiment_info.get('total_test_dialogues')}")

    print("\n[2/4] 評価実行中...")
    eval_result = evaluate_experiment(experiment_data)

    if eval_result is None:
        print("評価に失敗しました。")
        return

    dialogue_results, aggregated, accuracy = eval_result

    print("\n[3/4] 結果表示中...")
    print_aggregated_summary(aggregated, accuracy)

    print("\n[4/4] 結果保存中...")
    evaluation_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = evaluation_output_dir / f"evaluation_{timestamp}.csv"
    save_aggregated_results(
        aggregated, accuracy, dialogue_results, experiment_info, output_path
    )

    print("\n✓ 評価完了！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="嗜好抽出結果の評価スクリプト")
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="評価対象の実験結果JSONファイル",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="評価結果CSVの出力ディレクトリ",
    )
    args = parser.parse_args()

    main(experiment_results_path=args.results, evaluation_output_dir=args.output_dir)
