"""正規化関数群

experiment_evaluate.pyから分離した正規化ロジック
"""


def split_combined_axis(combined_axis: str) -> tuple[str, str | None]:
    """
    combined_axisを分割してaxisとsub_axisに分ける
    
    Args:
        combined_axis: "axis__sub_axis" 形式の文字列
    
    Returns:
        (axis, sub_axis): タプル
    """
    if "__" in combined_axis:
        parts = combined_axis.split("__", 1)
        return parts[0], parts[1]
    else:
        return combined_axis, None


def normalize_sub_axis(sub_axis_value: str | None) -> str | None:
    """
    sub_axis値を正規化する
    GT: "aesthetic/sensory" <-> EP: "aesthetic_sensory"
    
    Args:
        sub_axis_value: sub_axis文字列
    
    Returns:
        正規化されたsub_axis値（スラッシュをアンダースコアに統一）
    """
    if sub_axis_value is None or sub_axis_value == "":
        return None
    
    normalized = sub_axis_value.lower().strip().replace("/", "_")
    return normalized


def normalize_context(context_value) -> set:
    """
    コンテキスト値を正規化する
    GT: "solo", "group" <-> EP: "social-solo", "social-group"
    GT: "Morning" <-> EP: "temporal-morning"
    GT: "Working/studying" <-> EP: "activity-working_studying"
    
    Args:
        context_value: リストまたは文字列
    
    Returns:
        正規化されたコンテキストのセット（小文字、プレフィックス除去）
    """
    if context_value is None:
        return set()
    
    def normalize_single_context(ctx):
        """単一のコンテキスト値を正規化"""
        ctx_lower = ctx.lower().strip()
        
        if ctx_lower == "none":
            return None
        
        # プレフィックスを除去（例: "social-solo" -> "solo"）
        if "-" in ctx_lower:
            parts = ctx_lower.split("-", 1)
            if len(parts) == 2:
                return parts[1].replace("_", "/")
        
        return ctx_lower.replace("/", "_")
    
    if isinstance(context_value, list):
        contexts = [normalize_single_context(c) for c in context_value if c]
        contexts = [c for c in contexts if c is not None]
        return set(contexts)
    elif isinstance(context_value, str):
        normalized = normalize_single_context(context_value)
        return set([normalized]) if normalized else set()
    
    return set()


def normalize_intensity(intensity_value: str | None) -> str | None:
    """
    intensity値を正規化する
    
    Args:
        intensity_value: "high", "mid", "medium", "low" など
    
    Returns:
        正規化されたintensity値
    """
    if intensity_value is None:
        return None
    
    intensity_lower = intensity_value.lower().strip()
    
    # "mid" と "medium" を統一
    if intensity_lower in ["mid", "medium"]:
        return "medium"
    
    return intensity_lower
