"""対話単位評価モジュール

各対話ごとにOptimal Matchingを行い、TP/FP/FN/P/R/F1を計算する
scikit-learnを使用して評価指標を計算
"""

from dataclasses import dataclass, field

from .matching import find_optimal_matching, get_unmatched_predictions
from .metrics import augment_with_ancestors, compute_f1
from .normalizers import normalize_sub_axis, normalize_context, normalize_intensity


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------

@dataclass
class MatchComparison:
    """1ペアの属性比較結果"""
    is_matched: bool          # Entity一致？
    is_axis_ok: bool = False
    is_sub_axis_ok: bool = False
    is_polarity_ok: bool = False
    is_intensity_ok: bool = False
    is_context_ok: bool = False
    is_perfect_ok: bool = False
    h_axis_gt: set = field(default_factory=set)    # 階層的評価用
    h_axis_pred: set = field(default_factory=set)   # 階層的評価用
    h_axis_partial_score: float = 0.0               # 部分点


@dataclass
class DialogueComparison:
    """1対話の全ペア比較結果"""
    dialogue_id: str
    n_gt: int
    n_pred: int
    comparisons: list[MatchComparison] = field(default_factory=list)
    n_unmatched_pred: int = 0  # FP（マッチなし予測）


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


# ---------------------------------------------------------------------------
# Phase 1: 比較結果の収集
# ---------------------------------------------------------------------------

def collect_comparisons(
    dialogue_id: str,
    ground_truths: list[dict],
    predictions: list[dict],
    matching_results: list[tuple[int, int | None, dict | None, float]],
) -> DialogueComparison:
    """
    マッチング結果を属性ごとの比較結果に変換する。
    
    ドメインロジック（正規化・属性比較）をここに集約し、
    後段の compute_scores は純粋な数値集計のみを担う。
    """
    unmatched_preds = get_unmatched_predictions(predictions, matching_results)
    comparisons = []
    
    for gt_idx, pred_idx, match, score in matching_results:
        gt = ground_truths[gt_idx]
        
        if match is None:
            # マッチなし（MISSING）→ Entity不一致
            gt_aug = augment_with_ancestors(
                gt.get("axis", ""), normalize_sub_axis(gt.get("sub_axis"))
            )
            comparisons.append(MatchComparison(
                is_matched=False,
                h_axis_gt=gt_aug,
            ))
            continue
        
        # Entity一致 → 各属性を比較
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
        
        is_perfect_ok = all([is_axis_ok, is_sub_axis_ok, is_polarity_ok, is_intensity_ok, is_context_ok])
        
        # 階層的評価用 set
        gt_aug = augment_with_ancestors(gt.get("axis", ""), gt_sub_axis)
        pred_aug = augment_with_ancestors(match.get("axis", ""), pred_sub_axis)
        
        # 部分点: axis一致=0.5, sub_axis一致=0.5
        h_partial = (0.5 if is_axis_ok else 0.0) + (0.5 if is_sub_axis_ok else 0.0)
        
        comparisons.append(MatchComparison(
            is_matched=True,
            is_axis_ok=is_axis_ok,
            is_sub_axis_ok=is_sub_axis_ok,
            is_polarity_ok=is_polarity_ok,
            is_intensity_ok=is_intensity_ok,
            is_context_ok=is_context_ok,
            is_perfect_ok=is_perfect_ok,
            h_axis_gt=gt_aug,
            h_axis_pred=pred_aug,
            h_axis_partial_score=h_partial,
        ))
    
    return DialogueComparison(
        dialogue_id=dialogue_id,
        n_gt=len(ground_truths),
        n_pred=len(predictions),
        comparisons=comparisons,
        n_unmatched_pred=len(unmatched_preds),
    )


# ---------------------------------------------------------------------------
# Phase 2: スコア計算
# ---------------------------------------------------------------------------

def _calc_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """TP/FP/FNからPrecision/Recall/F1を計算"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_scores(comp: DialogueComparison) -> DialogueResult:
    """
    比較結果から評価スコアを算出する。
    
    純粋な集計ロジックのみ。ドメイン知識（正規化等）はここに含まない。
    """
    result = DialogueResult(
        dialogue_id=comp.dialogue_id,
        n_gt=comp.n_gt,
        n_pred=comp.n_pred,
    )
    
    # --- カウント集計 ---
    h_gt_total = h_pred_total = h_intersection_total = 0
    
    for mc in comp.comparisons:
        if not mc.is_matched:
            result.entity_fn += 1
            result.axis_fn += 1
            result.sub_axis_fn += 1
            result.polarity_fn += 1
            result.intensity_fn += 1
            result.context_fn += 1
            result.perfect_fn += 1
            h_gt_total += len(mc.h_axis_gt)
            continue
        
        # Entity TP
        result.entity_tp += 1
        
        # 各属性の TP / FN
        for attr in ("axis", "sub_axis", "polarity", "intensity", "context", "perfect"):
            if getattr(mc, f"is_{attr}_ok"):
                setattr(result, f"{attr}_tp", getattr(result, f"{attr}_tp") + 1)
            else:
                setattr(result, f"{attr}_fn", getattr(result, f"{attr}_fn") + 1)
        
        result.h_axis_tp += mc.h_axis_partial_score
        
        # 階層的評価
        h_gt_total += len(mc.h_axis_gt)
        h_pred_total += len(mc.h_axis_pred)
        h_intersection_total += len(mc.h_axis_gt & mc.h_axis_pred)
    
    # --- FP ---
    result.entity_fp = comp.n_unmatched_pred
    fp = result.entity_fp
    
    # --- Precision / Recall / F1 ---
    result.entity_precision, result.entity_recall, result.entity_f1 = _calc_prf(
        result.entity_tp, result.entity_fp, result.entity_fn)
    
    for attr in ("axis", "sub_axis", "polarity", "intensity", "context", "perfect"):
        p, r, f = _calc_prf(getattr(result, f"{attr}_tp"), fp, getattr(result, f"{attr}_fn"))
        setattr(result, f"{attr}_precision", p)
        setattr(result, f"{attr}_recall", r)
        setattr(result, f"{attr}_f1", f)
    
    # --- 条件付き分類精度 ---
    result.n_matched = result.entity_tp
    if result.n_matched > 0:
        for attr in ("axis", "sub_axis", "polarity", "intensity", "context", "perfect"):
            setattr(result, f"{attr}_accuracy",
                    getattr(result, f"{attr}_tp") / result.n_matched)
        result.h_axis_accuracy = result.h_axis_tp / result.n_matched
    
    # --- 階層的嗜好軸 ---
    result.h_axis_gt_size = h_gt_total
    result.h_axis_pred_size = h_pred_total
    result.h_axis_intersection = h_intersection_total
    result.h_axis_precision = h_intersection_total / h_pred_total if h_pred_total > 0 else 0.0
    result.h_axis_recall = h_intersection_total / h_gt_total if h_gt_total > 0 else 0.0
    result.h_axis_f1 = compute_f1(result.h_axis_precision, result.h_axis_recall)
    
    return result


# ---------------------------------------------------------------------------
# 公開 API（既存のインタフェースを維持）
# ---------------------------------------------------------------------------

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
    if len(ground_truths) == 0:
        result = DialogueResult(dialogue_id=dialogue_id, n_gt=0, n_pred=len(predictions))
        result.entity_fp = len(predictions)
        return result
    
    matching_results = find_optimal_matching(ground_truths, predictions)
    comparison = collect_comparisons(dialogue_id, ground_truths, predictions, matching_results)
    return compute_scores(comparison)
