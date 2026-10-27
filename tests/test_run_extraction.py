"""モデル判定の回帰テスト。"""

import pytest

from experiments.run_extraction import is_ollama_model


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-oss:20b", True),
        ("gemma4:e4b", True),
        ("qwen3.8:27b", True),
        ("gpt-4o", False),
        ("gpt-6-luna", False),
    ],
)
def test_is_ollama_model(model, expected):
    assert is_ollama_model(model) is expected
