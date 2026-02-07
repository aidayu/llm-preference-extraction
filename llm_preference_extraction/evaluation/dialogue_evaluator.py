"""対話単位評価モジュール

各対話ごとにOptimal Matchingを行い、TP/FP/FN/P/R/F1を計算する
scikit-learnを使用して評価指標を計算
"""

from dataclasses import dataclass

from sklearn.metrics import precision_recall_fscore_support

from .matching import find_optimal_matching, get_unmatched_predictions
from .metrics import augment_with_ancestors, compute_f1
from .normalizers import normalize_sub_axis, normalize_context, normalize_intensity


@dataclass
class DialogueResult:
    """対話ごとの評価結果"""
    dialogue_id: str
    n_gt: int
    n_pred: int
    
    # Entity評価
    entity_tp: int = 0
    entity_fp: int = 0
    entity_fn: int = 0
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_f1: float = 0.0
    
    # Axis評価
    axis_tp: int = 0
    axis_fn: int = 0
    axis_precision: float = 0.0
    axis_recall: float = 0.0
    axis_f1: float = 0.0
    
    # Sub-Axis評価
    sub_axis_tp: int = 0
    sub_axis_fn: int = 0
    sub_axis_precision: float = 0.0
    sub_axis_recall: float = 0.0
    sub_axis_f1: float = 0.0
    
    # 階層的嗜好軸評価
    h_axis_gt_size: int = 0
    h_axis_pred_size: int = 0
    h_axis_intersection: int = 0
    h_axis_precision: float = 0.0
    h_axis_recall: float = 0.0
    h_axis_f1: float = 0.0
    
    # 階層的嗜好軸の条件付き精度用 (部分点方式: axis一致=0.5, sub_axis一致=0.5)
    h_axis_tp: float = 0.0  # 部分点の合計
    
    # Polarity評価
    polarity_tp: int = 0
    polarity_fn: int = 0
    polarity_precision: float = 0.0
    polarity_recall: float = 0.0
    polarity_f1: float = 0.0
    
    # Intensity評価
    intensity_tp: int = 0
    intensity_fn: int = 0
    intensity_precision: float = 0.0
    intensity_recall: float = 0.0
    intensity_f1: float = 0.0
    
    # Context評価
    context_tp: int = 0
    context_fn: int = 0
    context_precision: float = 0.0
    context_recall: float = 0.0
    context_f1: float = 0.0
    
    # Perfect Match評価
    perfect_tp: int = 0
    perfect_fn: int = 0
    perfect_precision: float = 0.0
    perfect_recall: float = 0.0
    perfect_f1: float = 0.0
    
    # マッチしたペア数（Accuracy計算用）
    n_matched: int = 0
    
    # 条件付き分類精度 (Matched内でのAccuracy)
    axis_accuracy: float = 0.0
    sub_axis_accuracy: float = 0.0
    h_axis_accuracy: float = 0.0  # 階層的軸 (axis + sub_axis 両方一致)
    polarity_accuracy: float = 0.0
    intensity_accuracy: float = 0.0
    context_accuracy: float = 0.0
    perfect_accuracy: float = 0.0


def _calc_prf_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """TP/FP/FNからPrecision/Recall/F1を計算（sklearn互換）"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_dialogue(
    dialogue_id: str,
    ground_truths: list[dict],
    predictions: list[dict],
) -> DialogueResult:
    """
    1対話の評価を行う
    
    Args:
        dialogue_id: 対話ID
        ground_truths: 正解の嗜好オブジェクトリスト
        predictions: 予測の嗜好オブジェクトリスト
    
    Returns:
        DialogueResult: 評価結果
    """
    n_gt = len(ground_truths)
    n_pred = len(predictions)
    
    result = DialogueResult(dialogue_id=dialogue_id, n_gt=n_gt, n_pred=n_pred)
    
    if n_gt == 0:
        result.entity_fp = n_pred
        return result
    
    # Optimal Matchingを実行
    matching_results = find_optimal_matching(ground_truths, predictions)
    unmatched_preds = get_unmatched_predictions(predictions, matching_results)
    
    # 属性ごとの正解/予測ラベルを収集（バイナリ形式）
    attr_matches = {
        "axis": [], "sub_axis": [], "polarity": [], 
        "intensity": [], "context": [], "perfect": []
    }
    
    # 階層的評価用
    h_gt_total = h_pred_total = h_intersection_total = 0
    
    for gt_idx, pred_idx, match, score in matching_results:
        gt = ground_truths[gt_idx]
        
        if match is None:
            # マッチなし（MISSING）→ すべてFN
            result.entity_fn += 1
            for key in attr_matches:
                attr_matches[key].append((1, 0))  # (gt=1, pred=0)
            
            gt_aug = augment_with_ancestors(gt.get("axis", ""), normalize_sub_axis(gt.get("sub_axis")))
            h_gt_total += len(gt_aug)
            continue
        
        # Entity一致
        result.entity_tp += 1
        
        # 属性評価
        gt_sub_axis = normalize_sub_axis(gt.get("sub_axis"))
        pred_sub_axis = normalize_sub_axis(match.get("sub_axis"))
        
        is_axis_ok = match.get("axis") == gt.get("axis")
        is_sub_axis_ok = gt_sub_axis == pred_sub_axis
        is_polarity_ok = match.get("polarity") == gt.get("polarity")
        
        gt_int = normalize_intensity(gt.get("intensity"))
        pred_int = normalize_intensity(match.get("intensity"))
        is_intensity_ok = gt_int and pred_int and gt_int == pred_int
        
        gt_ctx = normalize_context(gt.get("context", []))
        pred_ctx = normalize_context(match.get("context_tags", []))
        is_context_ok = (len(gt_ctx) == 0 and len(pred_ctx) == 0) or (len(gt_ctx & pred_ctx) > 0)
        
        is_perfect = all([is_axis_ok, is_sub_axis_ok, is_polarity_ok, is_intensity_ok, is_context_ok])
        
        # TPカウント更新
        if is_axis_ok: result.axis_tp += 1
        else: result.axis_fn += 1
        
        if is_sub_axis_ok: result.sub_axis_tp += 1
        else: result.sub_axis_fn += 1
        
        # 階層的嗜好軸の条件付き精度用 (部分点方式)
        # axis一致 = 0.5点, sub_axis一致 = 0.5点 (合計最大1.0点)
        h_axis_score = 0.0
        if is_axis_ok:
            h_axis_score += 0.5
        if is_sub_axis_ok:
            h_axis_score += 0.5
        result.h_axis_tp += h_axis_score
        
        if is_polarity_ok: result.polarity_tp += 1
        else: result.polarity_fn += 1
        
        if is_intensity_ok: result.intensity_tp += 1
        else: result.intensity_fn += 1
        
        if is_context_ok: result.context_tp += 1
        else: result.context_fn += 1
        
        if is_perfect: result.perfect_tp += 1
        else: result.perfect_fn += 1
        
        # 階層的評価
        gt_aug = augment_with_ancestors(gt.get("axis", ""), gt_sub_axis)
        pred_aug = augment_with_ancestors(match.get("axis", ""), pred_sub_axis)
        h_gt_total += len(gt_aug)
        h_pred_total += len(pred_aug)
        h_intersection_total += len(gt_aug & pred_aug)
    
    # FPカウント
    result.entity_fp = len(unmatched_preds)
    fp = result.entity_fp
    
    # Precision/Recall/F1を計算
    result.entity_precision, result.entity_recall, result.entity_f1 = _calc_prf_from_counts(
        result.entity_tp, result.entity_fp, result.entity_fn)
    
    result.axis_precision, result.axis_recall, result.axis_f1 = _calc_prf_from_counts(
        result.axis_tp, fp, result.axis_fn)
    
    result.sub_axis_precision, result.sub_axis_recall, result.sub_axis_f1 = _calc_prf_from_counts(
        result.sub_axis_tp, fp, result.sub_axis_fn)
    
    result.polarity_precision, result.polarity_recall, result.polarity_f1 = _calc_prf_from_counts(
        result.polarity_tp, fp, result.polarity_fn)
    
    result.intensity_precision, result.intensity_recall, result.intensity_f1 = _calc_prf_from_counts(
        result.intensity_tp, fp, result.intensity_fn)
    
    result.context_precision, result.context_recall, result.context_f1 = _calc_prf_from_counts(
        result.context_tp, fp, result.context_fn)
    
    result.perfect_precision, result.perfect_recall, result.perfect_f1 = _calc_prf_from_counts(
        result.perfect_tp, fp, result.perfect_fn)
    
    # 条件付き分類精度 (マッチしたペア内でのAccuracy)
    result.n_matched = result.entity_tp
    if result.n_matched > 0:
        result.axis_accuracy = result.axis_tp / result.n_matched
        result.sub_axis_accuracy = result.sub_axis_tp / result.n_matched
        result.h_axis_accuracy = result.h_axis_tp / result.n_matched
        result.polarity_accuracy = result.polarity_tp / result.n_matched
        result.intensity_accuracy = result.intensity_tp / result.n_matched
        result.context_accuracy = result.context_tp / result.n_matched
        result.perfect_accuracy = result.perfect_tp / result.n_matched
    
    # 階層的嗜好軸
    result.h_axis_gt_size = h_gt_total
    result.h_axis_pred_size = h_pred_total
    result.h_axis_intersection = h_intersection_total
    result.h_axis_precision = h_intersection_total / h_pred_total if h_pred_total > 0 else 0.0
    result.h_axis_recall = h_intersection_total / h_gt_total if h_gt_total > 0 else 0.0
    result.h_axis_f1 = compute_f1(result.h_axis_precision, result.h_axis_recall)
    
    return result
