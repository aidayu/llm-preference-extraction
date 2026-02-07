"""集計モジュール

対話ごとの評価結果をMicro/Macro/Weighted F1で集計する
numpy を使用して効率的に計算
"""

from dataclasses import dataclass

import numpy as np

from .dialogue_evaluator import DialogueResult


@dataclass
class AggregatedMetrics:
    """集計された評価指標"""
    metric_name: str
    
    # Micro-F1（全サンプル統合）
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1: float = 0.0
    
    # Macro-F1（対話ごとのF1の平均）
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    
    # Weighted-F1（GT数で重み付け）
    weighted_precision: float = 0.0
    weighted_recall: float = 0.0
    weighted_f1: float = 0.0
    
    # 統計情報
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    n_dialogues: int = 0


@dataclass
class AggregatedAccuracy:
    """集計された条件付き分類精度"""
    # 各属性のマクロ平均Accuracy (対話ごとのAccuracyの平均)
    axis_accuracy: float = 0.0
    sub_axis_accuracy: float = 0.0
    h_axis_accuracy: float = 0.0  # 階層的軸 (axis + sub_axis 両方一致)
    polarity_accuracy: float = 0.0
    intensity_accuracy: float = 0.0
    context_accuracy: float = 0.0
    perfect_accuracy: float = 0.0
    
    # Micro Accuracy (全ペア統合)
    axis_micro_accuracy: float = 0.0
    sub_axis_micro_accuracy: float = 0.0
    h_axis_micro_accuracy: float = 0.0  # 階層的軸 (axis + sub_axis 両方一致)
    polarity_micro_accuracy: float = 0.0
    intensity_micro_accuracy: float = 0.0
    context_micro_accuracy: float = 0.0
    perfect_micro_accuracy: float = 0.0
    
    total_matched: int = 0
    n_dialogues: int = 0


def _compute_f1(precision: float, recall: float) -> float:
    """F1スコアを計算"""
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def aggregate_metrics(
    results: list[DialogueResult],
    metric_name: str,
    tp_attr: str,
    fn_attr: str,
    fp_attr: str = "entity_fp",
) -> AggregatedMetrics:
    """
    Micro/Macro/Weighted F1を計算する（numpy使用）
    """
    n = len(results)
    agg = AggregatedMetrics(metric_name=metric_name, n_dialogues=n)
    
    if n == 0:
        return agg
    
    # numpy配列に変換して効率的に計算
    tp = np.array([getattr(r, tp_attr, 0) for r in results])
    fn = np.array([getattr(r, fn_attr, 0) for r in results])
    fp = np.array([getattr(r, fp_attr, 0) for r in results])
    n_gt = np.array([r.n_gt for r in results])
    
    # Micro-F1: 全サンプルを合計
    total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
    agg.total_tp, agg.total_fp, agg.total_fn = int(total_tp), int(total_fp), int(total_fn)
    
    agg.micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    agg.micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    agg.micro_f1 = _compute_f1(agg.micro_precision, agg.micro_recall)
    
    # 対話ごとのP/R/F1を計算
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        recall = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    
    # Macro-F1: GTがある対話のみで平均
    valid_mask = n_gt > 0
    if valid_mask.sum() > 0:
        agg.macro_precision = float(precision[valid_mask].mean())
        agg.macro_recall = float(recall[valid_mask].mean())
        agg.macro_f1 = float(f1[valid_mask].mean())
    
    # Weighted-F1: GT数で重み付け平均
    total_gt = n_gt.sum()
    if total_gt > 0:
        agg.weighted_precision = float((precision * n_gt).sum() / total_gt)
        agg.weighted_recall = float((recall * n_gt).sum() / total_gt)
        agg.weighted_f1 = float((f1 * n_gt).sum() / total_gt)
    
    return agg


def aggregate_hierarchical_metrics(results: list[DialogueResult]) -> AggregatedMetrics:
    """階層的嗜好軸評価のMicro/Macro/Weighted F1を計算する"""
    n = len(results)
    agg = AggregatedMetrics(metric_name="Hierarchical Axis", n_dialogues=n)
    
    if n == 0:
        return agg
    
    # numpy配列に変換
    gt_size = np.array([r.h_axis_gt_size for r in results])
    pred_size = np.array([r.h_axis_pred_size for r in results])
    intersection = np.array([r.h_axis_intersection for r in results])
    n_gt = np.array([r.n_gt for r in results])
    
    # Micro-F1
    total_gt_size, total_pred_size = gt_size.sum(), pred_size.sum()
    total_intersection = intersection.sum()
    
    agg.micro_precision = total_intersection / total_pred_size if total_pred_size > 0 else 0.0
    agg.micro_recall = total_intersection / total_gt_size if total_gt_size > 0 else 0.0
    agg.micro_f1 = _compute_f1(agg.micro_precision, agg.micro_recall)
    
    # 対話ごとのP/R/F1
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(pred_size > 0, intersection / pred_size, 0.0)
        recall = np.where(gt_size > 0, intersection / gt_size, 0.0)
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    
    # Macro/Weighted
    valid_mask = n_gt > 0
    if valid_mask.sum() > 0:
        agg.macro_precision = float(precision[valid_mask].mean())
        agg.macro_recall = float(recall[valid_mask].mean())
        agg.macro_f1 = float(f1[valid_mask].mean())
    
    total_gt_items = n_gt.sum()
    if total_gt_items > 0:
        agg.weighted_precision = float((precision * n_gt).sum() / total_gt_items)
        agg.weighted_recall = float((recall * n_gt).sum() / total_gt_items)
        agg.weighted_f1 = float((f1 * n_gt).sum() / total_gt_items)
    
    return agg


def aggregate_accuracy(results: list[DialogueResult]) -> AggregatedAccuracy:
    """条件付き分類精度を集計する"""
    n = len(results)
    agg = AggregatedAccuracy(n_dialogues=n)
    
    if n == 0:
        return agg
    
    # numpy配列に変換
    n_matched = np.array([r.n_matched for r in results])
    axis_tp = np.array([r.axis_tp for r in results])
    sub_axis_tp = np.array([r.sub_axis_tp for r in results])
    h_axis_tp = np.array([r.h_axis_tp for r in results])
    polarity_tp = np.array([r.polarity_tp for r in results])
    intensity_tp = np.array([r.intensity_tp for r in results])
    context_tp = np.array([r.context_tp for r in results])
    perfect_tp = np.array([r.perfect_tp for r in results])
    
    # Micro Accuracy: 全マッチペア統合
    total_matched = n_matched.sum()
    agg.total_matched = int(total_matched)
    
    if total_matched > 0:
        agg.axis_micro_accuracy = float(axis_tp.sum() / total_matched)
        agg.sub_axis_micro_accuracy = float(sub_axis_tp.sum() / total_matched)
        agg.h_axis_micro_accuracy = float(h_axis_tp.sum() / total_matched)
        agg.polarity_micro_accuracy = float(polarity_tp.sum() / total_matched)
        agg.intensity_micro_accuracy = float(intensity_tp.sum() / total_matched)
        agg.context_micro_accuracy = float(context_tp.sum() / total_matched)
        agg.perfect_micro_accuracy = float(perfect_tp.sum() / total_matched)
    
    # Macro Accuracy: 対話ごとのAccuracyの平均 (マッチありの対話のみ)
    valid_mask = n_matched > 0
    if valid_mask.sum() > 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            axis_acc = np.where(n_matched > 0, axis_tp / n_matched, 0.0)
            sub_axis_acc = np.where(n_matched > 0, sub_axis_tp / n_matched, 0.0)
            h_axis_acc = np.where(n_matched > 0, h_axis_tp / n_matched, 0.0)
            polarity_acc = np.where(n_matched > 0, polarity_tp / n_matched, 0.0)
            intensity_acc = np.where(n_matched > 0, intensity_tp / n_matched, 0.0)
            context_acc = np.where(n_matched > 0, context_tp / n_matched, 0.0)
            perfect_acc = np.where(n_matched > 0, perfect_tp / n_matched, 0.0)
        
        agg.axis_accuracy = float(axis_acc[valid_mask].mean())
        agg.sub_axis_accuracy = float(sub_axis_acc[valid_mask].mean())
        agg.h_axis_accuracy = float(h_axis_acc[valid_mask].mean())
        agg.polarity_accuracy = float(polarity_acc[valid_mask].mean())
        agg.intensity_accuracy = float(intensity_acc[valid_mask].mean())
        agg.context_accuracy = float(context_acc[valid_mask].mean())
        agg.perfect_accuracy = float(perfect_acc[valid_mask].mean())
    
    return agg


def aggregate_all_metrics(results: list[DialogueResult]) -> tuple[dict[str, AggregatedMetrics], AggregatedAccuracy]:
    """すべての評価指標を集計する"""
    f1_metrics = {
        "Entity": aggregate_metrics(results, "Entity", "entity_tp", "entity_fn", "entity_fp"),
        "Axis": aggregate_metrics(results, "Axis", "axis_tp", "axis_fn", "entity_fp"),
        "Sub-Axis": aggregate_metrics(results, "Sub-Axis", "sub_axis_tp", "sub_axis_fn", "entity_fp"),
        "Hierarchical Axis": aggregate_hierarchical_metrics(results),
        "Polarity": aggregate_metrics(results, "Polarity", "polarity_tp", "polarity_fn", "entity_fp"),
        "Intensity": aggregate_metrics(results, "Intensity", "intensity_tp", "intensity_fn", "entity_fp"),
        "Context": aggregate_metrics(results, "Context", "context_tp", "context_fn", "entity_fp"),
        "Perfect Match": aggregate_metrics(results, "Perfect Match", "perfect_tp", "perfect_fn", "entity_fp"),
    }
    accuracy_metrics = aggregate_accuracy(results)
    return f1_metrics, accuracy_metrics
