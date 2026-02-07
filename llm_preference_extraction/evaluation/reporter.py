"""評価結果出力モジュール

CSV保存、サマリー表示のロジック
"""

import csv
from datetime import datetime


def save_evaluation_results(metrics: dict, experiment_info: dict, output_path: str):
    """
    評価結果をCSVファイルに保存する
    
    Args:
        metrics: 評価指標
        experiment_info: 実験メタデータ
        output_path: 出力ファイルパス
    """
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        
        # 実験情報
        writer.writerow(["Experiment Information", "", ""])
        writer.writerow(["Timestamp", experiment_info.get("timestamp", ""), ""])
        writer.writerow(["Model", experiment_info.get("model", ""), ""])
        writer.writerow(["Few-shot IDs", str(experiment_info.get("few_shot_ids", [])), ""])
        writer.writerow(["Total Test Dialogues", experiment_info.get("total_test_dialogues", ""), ""])
        writer.writerow(["", "", ""])
        
        # 評価結果
        writer.writerow(["Metric", "Value", "Count"])
        writer.writerow(["Total Ground Truth Annotations", metrics["total_gt_items"], ""])
        writer.writerow(["Total LLM Predictions", metrics["total_pred_items"], ""])
        writer.writerow(["Matched Entities", metrics["matched_items"], ""])
        writer.writerow(["", "", ""])
        
        # Entity-level metrics
        writer.writerow(["Entity Recall", f"{metrics['entity_recall']:.2%}", f"{metrics['matched_items']}/{metrics['total_gt_items']}"])
        writer.writerow(["Entity Precision", f"{metrics['entity_precision']:.2%}", f"{metrics['matched_items']}/{metrics['total_pred_items']}"])
        writer.writerow(["Entity F1", f"{metrics['entity_f1']:.2%}", ""])
        writer.writerow(["", "", ""])
        
        # Axis metrics
        writer.writerow(["Axis Accuracy", f"{metrics['axis_accuracy']:.2%}", f"{metrics['correct_axis']}/{metrics['total_gt_items']}"])
        writer.writerow(["Axis Recall", f"{metrics['axis_recall']:.2%}", f"{metrics['correct_axis']}/{metrics['total_gt_items']}"])
        writer.writerow(["Axis Precision", f"{metrics['axis_precision']:.2%}", ""])
        writer.writerow(["Axis F1", f"{metrics['axis_f1']:.2%}", ""])
        writer.writerow(["", "", ""])
        
        # Sub-Axis metrics
        writer.writerow(["Sub-Axis Accuracy", f"{metrics['sub_axis_accuracy']:.2%}", f"{metrics['correct_sub_axis']}/{metrics['total_gt_items']}"])
        writer.writerow(["Sub-Axis Recall", f"{metrics['sub_axis_recall']:.2%}", f"{metrics['correct_sub_axis']}/{metrics['total_gt_items']}"])
        writer.writerow(["Sub-Axis Precision", f"{metrics['sub_axis_precision']:.2%}", ""])
        writer.writerow(["Sub-Axis F1", f"{metrics['sub_axis_f1']:.2%}", ""])
        writer.writerow(["", "", ""])
        
        # Hierarchical Axis metrics
        writer.writerow(["Hierarchical Axis Recall", f"{metrics['h_axis_recall']:.2%}", f"{metrics['h_intersection_size']}/{metrics['h_gt_augmented_size']}"])
        writer.writerow(["Hierarchical Axis Precision", f"{metrics['h_axis_precision']:.2%}", f"{metrics['h_intersection_size']}/{metrics['h_pred_augmented_size']}"])
        writer.writerow(["Hierarchical Axis F1", f"{metrics['h_axis_f1']:.2%}", ""])
        writer.writerow(["", "", ""])
        
        # Polarity metrics
        writer.writerow(["Polarity Accuracy", f"{metrics['polarity_accuracy']:.2%}", f"{metrics['correct_polarity']}/{metrics['total_gt_items']}"])
        writer.writerow(["Polarity Recall", f"{metrics['polarity_recall']:.2%}", f"{metrics['correct_polarity']}/{metrics['total_gt_items']}"])
        writer.writerow(["Polarity Precision", f"{metrics['polarity_precision']:.2%}", ""])
        writer.writerow(["Polarity F1", f"{metrics['polarity_f1']:.2%}", ""])
        writer.writerow(["", "", ""])
        
        # Intensity metrics
        writer.writerow(["Intensity Accuracy", f"{metrics['intensity_accuracy']:.2%}", f"{metrics['correct_intensity']}/{metrics['total_gt_items']}"])
        writer.writerow(["Intensity Recall", f"{metrics['intensity_recall']:.2%}", f"{metrics['correct_intensity']}/{metrics['total_gt_items']}"])
        writer.writerow(["Intensity Precision", f"{metrics['intensity_precision']:.2%}", ""])
        writer.writerow(["Intensity F1", f"{metrics['intensity_f1']:.2%}", ""])
        writer.writerow(["", "", ""])
        
        # Context metrics
        writer.writerow(["Context Accuracy", f"{metrics['context_accuracy']:.2%}", f"{metrics['correct_context']}/{metrics['total_gt_items']}"])
        writer.writerow(["Context Recall", f"{metrics['context_recall']:.2%}", f"{metrics['correct_context']}/{metrics['total_gt_items']}"])
        writer.writerow(["Context Precision", f"{metrics['context_precision']:.2%}", ""])
        writer.writerow(["Context F1", f"{metrics['context_f1']:.2%}", ""])
        writer.writerow(["", "", ""])
        
        # Perfect match metrics
        writer.writerow(["Perfect Match Accuracy", f"{metrics['perfect_match_accuracy']:.2%}", f"{metrics['perfect_matches']}/{metrics['total_gt_items']}"])
        writer.writerow(["Perfect Match Recall", f"{metrics['perfect_recall']:.2%}", f"{metrics['perfect_matches']}/{metrics['total_gt_items']}"])
        writer.writerow(["Perfect Match Precision", f"{metrics['perfect_precision']:.2%}", ""])
        writer.writerow(["Perfect Match F1", f"{metrics['perfect_f1']:.2%}", ""])
    
    print(f"\n評価結果を保存: {output_path}")


def print_evaluation_summary(metrics: dict):
    """評価指標のサマリーを表示する"""
    if not metrics:
        return
    
    print("\n" + "="*80)
    print("=== 評価結果サマリー ===")
    print("="*80)
    print(f"Total Ground Truth: {metrics['total_gt_items']}")
    print(f"Total Predictions:  {metrics['total_pred_items']}")
    print(f"Matched Entities:   {metrics['matched_items']}")
    print()
    
    print("--- Entity-level Metrics ---")
    print(f"  Recall:    {metrics['entity_recall']:.2%} ({metrics['matched_items']}/{metrics['total_gt_items']})")
    print(f"  Precision: {metrics['entity_precision']:.2%} ({metrics['matched_items']}/{metrics['total_pred_items']})")
    print(f"  F1:        {metrics['entity_f1']:.2%}")
    print()
    
    print("--- Axis Metrics ---")
    print(f"  Accuracy:  {metrics['axis_accuracy']:.2%} ({metrics['correct_axis']}/{metrics['total_gt_items']})")
    print(f"  Recall:    {metrics['axis_recall']:.2%}")
    print(f"  Precision: {metrics['axis_precision']:.2%}")
    print(f"  F1:        {metrics['axis_f1']:.2%}")
    print()
    
    print("--- Sub-Axis Metrics ---")
    print(f"  Accuracy:  {metrics['sub_axis_accuracy']:.2%} ({metrics['correct_sub_axis']}/{metrics['total_gt_items']})")
    print(f"  Recall:    {metrics['sub_axis_recall']:.2%}")
    print(f"  Precision: {metrics['sub_axis_precision']:.2%}")
    print(f"  F1:        {metrics['sub_axis_f1']:.2%}")
    print()
    
    print("--- Hierarchical Axis Metrics (Kiritchenko et al. 2006) ---")
    print(f"  hRecall:    {metrics['h_axis_recall']:.2%} ({metrics['h_intersection_size']}/{metrics['h_gt_augmented_size']})")
    print(f"  hPrecision: {metrics['h_axis_precision']:.2%} ({metrics['h_intersection_size']}/{metrics['h_pred_augmented_size']})")
    print(f"  hF1:        {metrics['h_axis_f1']:.2%}")
    print()
    
    print("--- Polarity Metrics ---")
    print(f"  Accuracy:  {metrics['polarity_accuracy']:.2%} ({metrics['correct_polarity']}/{metrics['total_gt_items']})")
    print(f"  Recall:    {metrics['polarity_recall']:.2%}")
    print(f"  Precision: {metrics['polarity_precision']:.2%}")
    print(f"  F1:        {metrics['polarity_f1']:.2%}")
    print()
    
    print("--- Intensity Metrics ---")
    print(f"  Accuracy:  {metrics['intensity_accuracy']:.2%} ({metrics['correct_intensity']}/{metrics['total_gt_items']})")
    print(f"  Recall:    {metrics['intensity_recall']:.2%}")
    print(f"  Precision: {metrics['intensity_precision']:.2%}")
    print(f"  F1:        {metrics['intensity_f1']:.2%}")
    print()
    
    print("--- Context Metrics ---")
    print(f"  Accuracy:  {metrics['context_accuracy']:.2%} ({metrics['correct_context']}/{metrics['total_gt_items']})")
    print(f"  Recall:    {metrics['context_recall']:.2%}")
    print(f"  Precision: {metrics['context_precision']:.2%}")
    print(f"  F1:        {metrics['context_f1']:.2%}")
    print()
    
    print("--- Perfect Match Metrics (All Fields) ---")
    print(f"  Accuracy:  {metrics['perfect_match_accuracy']:.2%} ({metrics['perfect_matches']}/{metrics['total_gt_items']})")
    print(f"  Recall:    {metrics['perfect_recall']:.2%}")
    print(f"  Precision: {metrics['perfect_precision']:.2%}")
    print(f"  F1:        {metrics['perfect_f1']:.2%}")
    print("="*80)
