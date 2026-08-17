"""cost_estimator の回帰テスト。

トークン実測は実ファイル（プロンプト・データセット）に依存するため、
ここでは「単価計算」「回数スケール」「ローカル無料」「キャッシュ割引」を固定値で検証する。
"""

from __future__ import annotations

from cost_estimator.estimator import (
    DEFAULT_DATASET_FOR_TEST,
    estimate,
    load_pricing,
)

PRICING = load_pricing()


def _est(model: str, runs: int = 1, output_tokens: int = 500):
    return estimate(
        model=model,
        dataset_path=DEFAULT_DATASET_FOR_TEST,
        few_shot_ids=[0, 1, 2],
        pricing=PRICING,
        output_tokens_per_call=output_tokens,
        runs=runs,
    )


def test_terra_is_input_same_output_higher_than_4o():
    """GPT-5.6 Terra は入力単価が gpt-4o と同額、出力のみ1.5倍。"""
    terra = PRICING["models"]["gpt-5.6-terra"]
    gpt4o = PRICING["models"]["gpt-4o"]
    assert terra["input_per_mtok"] == gpt4o["input_per_mtok"]
    assert terra["output_per_mtok"] == gpt4o["output_per_mtok"] * 1.5


def test_cost_scales_linearly_with_runs():
    one = _est("gpt-5.6-terra", runs=1)
    three = _est("gpt-5.6-terra", runs=3)
    assert three.calls == one.calls * 3
    assert three.input_tokens == one.input_tokens * 3
    assert abs(three.total_cost - one.total_cost * 3) < 1e-9


def test_local_model_is_free():
    est = _est("gemma4:31b")
    assert est.is_local
    assert est.total_cost == 0.0


def test_fixed_prompt_dominates_input():
    """few-shot system + schema が入力の大半を占める（キャッシュ割引が効く根拠）。"""
    est = _est("gpt-5.6-terra")
    fixed = est.calls * (est.system_tokens + est.schema_tokens)
    assert fixed / est.input_tokens > 0.9


def test_cached_cost_is_cheaper_and_only_discounts_input():
    est = _est("gpt-5.6-terra")
    assert est.cached_total_cost is not None
    assert est.cached_total_cost < est.total_cost
    # 出力コストは割引対象外なので変わらない
    assert abs(est.cached_total_cost - est.cached_input_cost - est.output_cost) < 1e-9


def test_gpt4o_has_no_cache_discount():
    """gpt-4o は pricing.yaml でキャッシュ割引を定義していない。"""
    est = _est("gpt-4o")
    assert est.cached_total_cost is None


def test_unknown_model_raises():
    import pytest

    with pytest.raises(KeyError):
        _est("gpt-nonexistent")
