"""マッチングモジュール

GT（正解）とPred（予測）の嗜好オブジェクト間の最適マッチングを行う。
Hungarian algorithmを使用して、スコア合計が最大となる1対1マッチングを計算する。
意味的類似度計算にはOpenAI Embeddingsを使用する。
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .normalizers import normalize_sub_axis, normalize_context, normalize_intensity
from .semantic_similarity import compute_entity_similarity, ENTITY_SIMILARITY_THRESHOLD


def compute_matching_score(gt: dict, pred: dict) -> float:
    """
    GTとPred嗜好オブジェクト間のマッチングスコアを計算する。
    
    スコア計算式:
        スコア = entity類似度 × 10 + axis一致(1 or 0) + sub_axis一致(1 or 0)
        
        - entity類似度が閾値未満の場合はスコア0（マッチ不可）
        - sub_axis一致はaxisが一致している場合のみ考慮
    
    Args:
        gt: 正解の嗜好オブジェクト
        pred: 予測の嗜好オブジェクト
    
    Returns:
        0.0-12.0のスコア（高いほど良いマッチ）
    """
    ENTITY_WEIGHT = 10
    
    # Entity意味的類似度チェック（必須条件）
    gt_entity = gt.get("entity", "").strip()
    pred_entity = pred.get("entity", "").strip()
    
    if not gt_entity or not pred_entity:
        return 0.0
    
    entity_similarity = compute_entity_similarity(gt_entity, pred_entity)
    
    if entity_similarity < ENTITY_SIMILARITY_THRESHOLD:
        # 類似度が閾値未満の場合はマッチ不可
        return 0.0
    
    # スコア計算開始
    score = entity_similarity * ENTITY_WEIGHT
    
    # Axis一致
    axis_match = gt.get("axis") == pred.get("axis")
    if axis_match:
        score += 1.0
    
    # Sub-axis一致（Axisが一致している場合のみ）
    if axis_match:
        gt_sub_axis = normalize_sub_axis(gt.get("sub_axis"))
        pred_sub_axis = normalize_sub_axis(pred.get("sub_axis"))
        if gt_sub_axis == pred_sub_axis:
            score += 1.0
    
    return score



def find_optimal_matching(
    ground_truths: list[dict],
    predictions: list[dict],
) -> list[tuple[int, int | None, dict | None, float]]:
    """
    Hungarian algorithmを使用してGTとPred間の最適マッチングを見つける。
    
    Args:
        ground_truths: 正解の嗜好オブジェクトリスト
        predictions: 予測の嗜好オブジェクトリスト
    
    Returns:
        マッチング結果のリスト: [(gt_idx, pred_idx or None, pred or None, score), ...]
        pred_idx=None の場合はマッチなし（MISSING）
    """
    n_gt = len(ground_truths)
    n_pred = len(predictions)
    
    if n_gt == 0:
        return []
    
    if n_pred == 0:
        # 予測がない場合、すべてのGTはマッチなし
        return [(i, None, None, 0.0) for i in range(n_gt)]
    
    # スコア行列を作成（GTがrow, Predがcolumn）
    score_matrix = np.zeros((n_gt, n_pred))
    
    for i, gt in enumerate(ground_truths):
        for j, pred in enumerate(predictions):
            score_matrix[i, j] = compute_matching_score(gt, pred)
    
    # Hungarian algorithmはコスト最小化なので、スコアを負にする
    cost_matrix = -score_matrix
    
    # 最適割り当てを計算
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 結果を構築
    results = []
    matched_gt_indices = set()
    
    for gt_idx, pred_idx in zip(row_ind, col_ind):
        score = score_matrix[gt_idx, pred_idx]
        
        if score > 0:  # entity一致がある場合のみマッチとみなす
            results.append((gt_idx, pred_idx, predictions[pred_idx], score))
            matched_gt_indices.add(gt_idx)
        else:
            results.append((gt_idx, None, None, 0.0))
            matched_gt_indices.add(gt_idx)
    
    # マッチングに含まれなかったGTを追加
    for i in range(n_gt):
        if i not in matched_gt_indices:
            results.append((i, None, None, 0.0))
    
    # GT indexでソート
    results.sort(key=lambda x: x[0])
    
    return results


def get_unmatched_predictions(
    predictions: list[dict],
    matching_results: list[tuple[int, int | None, dict | None, float]],
) -> list[tuple[int, dict]]:
    """
    マッチングされなかった予測を取得する（False Positive検出用）。
    
    Args:
        predictions: 予測の嗜好オブジェクトリスト
        matching_results: find_optimal_matchingの結果
    
    Returns:
        マッチされなかった予測: [(pred_idx, pred), ...]
    """
    matched_pred_indices = {
        result[1] for result in matching_results if result[1] is not None
    }
    
    return [
        (i, pred) for i, pred in enumerate(predictions)
        if i not in matched_pred_indices
    ]
