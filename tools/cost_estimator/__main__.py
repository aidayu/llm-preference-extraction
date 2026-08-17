"""抽出実験のコスト見積もり CLI。

    uv run python -m cost_estimator --model gpt-5.6-terra
    uv run python -m cost_estimator --model gpt-5.6-terra,gpt-4o,gpt-4o-mini --full
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cost_estimator.estimator import (
    PROJECT_ROOT,
    Estimate,
    estimate,
    load_pricing,
    measure_output_tokens,
)

DEFAULT_DATASET = PROJECT_ROOT / "data" / "ground_truth" / "test.json"
FULL_DATASET = PROJECT_ROOT / "data" / "ground_truth" / "dailydialog_annotated_integrated.json"
DEFAULT_FEW_SHOT_IDS = [0, 1, 2]


def format_row(est: Estimate) -> str:
    cost = "(local)" if est.is_local else f"{est.total_cost:8.4f}"
    return (
        f"{est.model:<16} {est.calls:>6} {est.input_tokens:>11,} "
        f"{est.output_tokens:>10,} {cost:>10}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="抽出実験の API コスト見積もり")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.6-terra",
        help="モデル名のカンマ区切り（pricing.yaml に定義があるもの）",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=f"データセットパス（既定: {DEFAULT_DATASET.name}）",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=f"50件版データセット（{FULL_DATASET.name}）を使う",
    )
    parser.add_argument(
        "--few-shot-ids",
        type=str,
        default=",".join(map(str, DEFAULT_FEW_SHOT_IDS)),
        help="Few-shot に使う dialogue_id のカンマ区切り",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=None,
        help="1コールあたり出力トークン（省略時は過去実行結果から実測）",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="同一データセットを流す回数（反復実験用）",
    )
    args = parser.parse_args()

    dataset_path = args.dataset or (FULL_DATASET if args.full else DEFAULT_DATASET)
    few_shot_ids = [int(x) for x in args.few_shot_ids.split(",") if x.strip()]
    models = [m.strip() for m in args.model.split(",") if m.strip()]

    pricing = load_pricing()

    measured = args.output_tokens
    source = "指定値"
    if measured is None:
        measured = measure_output_tokens()
        source = "過去実行結果からの実測平均" if measured else "フォールバック既定値"

    estimates = [
        estimate(
            model=m,
            dataset_path=dataset_path,
            few_shot_ids=few_shot_ids,
            pricing=pricing,
            output_tokens_per_call=args.output_tokens,
            runs=args.runs,
        )
        for m in models
    ]

    head = estimates[0]
    print(f"\nデータセット: {dataset_path.name}")
    print(f"Few-shot ids: {few_shot_ids}（テスト対象から除外）")
    print(f"テスト対話数: {head.calls // args.runs} × runs={args.runs} = {head.calls} コール")
    print(
        f"1コールあたり入力: system {head.system_tokens:,} + schema {head.schema_tokens:,} "
        f"(固定) + user 平均 {head.user_tokens // head.calls:,}"
    )
    print(f"1コールあたり出力: {head.output_tokens_per_call:,} tok（{source}）")
    print()

    print(f"{'model':<16} {'calls':>6} {'in_tok':>11} {'out_tok':>10} {'cost($)':>10}")
    print("-" * 58)
    for est in estimates:
        print(format_row(est))
    print()

    cached = [e for e in estimates if e.cached_total_cost is not None]
    if cached:
        print("プロンプトキャッシュ有効時（system+schema が2コール目以降キャッシュヒット）:")
        for est in cached:
            saving = est.total_cost - est.cached_total_cost
            pct = saving / est.total_cost * 100 if est.total_cost else 0
            print(
                f"  {est.model:<16} {est.cached_total_cost:8.4f}  "
                f"(-{saving:.4f}, -{pct:.0f}%)"
            )
        print()


if __name__ == "__main__":
    main()
