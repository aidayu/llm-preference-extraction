"""評価指標計算モジュール

階層的F1スコア、各種Precision/Recall/F1の計算ロジック
"""

from .normalizers import normalize_sub_axis


def augment_with_ancestors(axis: str, sub_axis: str | None) -> set:
    """
    予測または正解のラベルを親を含めて拡張する
    
    階層: axis (親) -> sub_axis (子)
    例: liking, aesthetic_sensory -> {"liking", "liking__aesthetic_sensory"}
    
    Args:
        axis: 親軸 (liking, wanting, need)
        sub_axis: 子軸 (aesthetic_sensory, goal など)
    
    Returns:
        拡張されたラベルセット
    """
    labels = set()
    
    if axis:
        labels.add(axis)
        if sub_axis:
            labels.add(f"{axis}__{sub_axis}")
    
    return labels


def compute_hierarchical_metrics(gt_pairs: list, pred_pairs: list) -> dict:
    """
    階層的Precision/Recall/F1を計算する (Micro平均)
    
    Kiritchenko et al. (2006) の手法:
    - 各ペアごとに予測と正解を親ラベルで拡張 (augment)
    - 全ペアのラベルを集約（重複許容）
    - hP = Σ|Y_aug ∩ Ŷ_aug| / Σ|Ŷ_aug|
    - hR = Σ|Y_aug ∩ Ŷ_aug| / Σ|Y_aug|
    - hF1 = 2 * hP * hR / (hP + hR)
    
    Args:
        gt_pairs: [(axis, sub_axis), ...] 正解ラベルのリスト
        pred_pairs: [(axis, sub_axis), ...] 予測ラベルのリスト（同じ長さ）
    
    Returns:
        {"h_precision": float, "h_recall": float, "h_f1": float, ...}
    """
    if len(gt_pairs) != len(pred_pairs):
        raise ValueError("gt_pairs and pred_pairs must have the same length")
    
    total_gt_labels = 0
    total_pred_labels = 0
    total_intersection = 0
    
    for (gt_axis, gt_sub_axis), (pred_axis, pred_sub_axis) in zip(gt_pairs, pred_pairs):
        gt_aug = augment_with_ancestors(gt_axis, gt_sub_axis)
        pred_aug = augment_with_ancestors(pred_axis, pred_sub_axis)
        
        intersection = gt_aug & pred_aug
        
        total_gt_labels += len(gt_aug)
        total_pred_labels += len(pred_aug)
        total_intersection += len(intersection)
    
    h_precision = total_intersection / total_pred_labels if total_pred_labels > 0 else 0
    h_recall = total_intersection / total_gt_labels if total_gt_labels > 0 else 0
    h_f1 = (2 * h_precision * h_recall) / (h_precision + h_recall) if (h_precision + h_recall) > 0 else 0
    
    return {
        "h_precision": h_precision,
        "h_recall": h_recall,
        "h_f1": h_f1,
        "gt_augmented_size": total_gt_labels,
        "pred_augmented_size": total_pred_labels,
        "intersection_size": total_intersection,
    }


def compute_f1(precision: float, recall: float) -> float:
    """F1スコアを計算"""
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def compute_basic_metrics(
    total_gt: int,
    total_pred: int, 
    correct_count: int,
    tp_count: int
) -> dict:
    """
    基本的なPrecision/Recall/F1を計算
    
    Args:
        total_gt: 正解総数
        total_pred: 予測総数
        correct_count: 正解数（GTベース）
        tp_count: True Positive数（Predベース）
    
    Returns:
        {"accuracy": float, "recall": float, "precision": float, "f1": float}
    """
    recall = correct_count / total_gt if total_gt > 0 else 0
    precision = tp_count / total_pred if total_pred > 0 else 0
    f1 = compute_f1(precision, recall)
    
    return {
        "accuracy": recall,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }
