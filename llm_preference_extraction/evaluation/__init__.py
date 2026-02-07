"""評価モジュール

嗜好抽出結果の評価に関する機能を提供
- normalizers: 正規化関数
- metrics: 評価指標計算
- matching: GT-Pred間の最適マッチング
- semantic_similarity: 意味的類似度計算（OpenAI Embeddings）
- dialogue_evaluator: 対話単位評価
- aggregators: Micro/Macro/Weighted F1集計
- reporter: 結果出力
"""

from .normalizers import (
    split_combined_axis,
    normalize_sub_axis,
    normalize_context,
    normalize_intensity,
)
from .metrics import (
    augment_with_ancestors,
    compute_hierarchical_metrics,
    compute_f1,
    compute_basic_metrics,
)
from .matching import (
    compute_matching_score,
    find_optimal_matching,
    get_unmatched_predictions,
)
from .semantic_similarity import (
    compute_entity_similarity,
    entities_match,
    clear_embedding_cache,
    get_cache_info,
    ENTITY_SIMILARITY_THRESHOLD,
)
from .dialogue_evaluator import (
    DialogueResult,
    evaluate_dialogue,
)
from .aggregators import (
    AggregatedMetrics,
    AggregatedAccuracy,
    aggregate_metrics,
    aggregate_accuracy,
    aggregate_hierarchical_metrics,
    aggregate_all_metrics,
)
from .reporter import (
    save_evaluation_results,
    print_evaluation_summary,
)


__all__ = [
    # normalizers
    "split_combined_axis",
    "normalize_sub_axis", 
    "normalize_context",
    "normalize_intensity",
    # metrics
    "augment_with_ancestors",
    "compute_hierarchical_metrics",
    "compute_f1",
    "compute_basic_metrics",
    # matching
    "compute_matching_score",
    "find_optimal_matching",
    "get_unmatched_predictions",
    # semantic_similarity
    "compute_entity_similarity",
    "entities_match",
    "clear_embedding_cache",
    "get_cache_info",
    "ENTITY_SIMILARITY_THRESHOLD",
    # dialogue_evaluator
    "DialogueResult",
    "evaluate_dialogue",
    # aggregators
    "AggregatedMetrics",
    "AggregatedAccuracy",
    "aggregate_metrics",
    "aggregate_accuracy",
    "aggregate_hierarchical_metrics",
    "aggregate_all_metrics",
    # reporter
    "save_evaluation_results",
    "print_evaluation_summary",
]

