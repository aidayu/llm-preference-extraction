"""モデル判定の回帰テスト。"""

import pytest

from llm_preference_extraction.extractors import supports_temperature


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-6-luna", False),
        ("gpt-5.2", False),
        ("gpt-4o", True),
        ("gemma4:e4b", True),
    ],
)
def test_supports_temperature(model, expected):
    assert supports_temperature(model) is expected
