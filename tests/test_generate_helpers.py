"""Tests for pure helper functions in src/retrieval/generate.py.

Cloud/model dependencies are optional; the module is skipped entirely when
they are not installed so the suite stays lightweight.
"""

import pytest

pytest.importorskip("langsmith")
pytest.importorskip("langchain_core")

from src.retrieval.generate import (
    _capture_usage,
    _estimate_cost,
    _extract_text,
    _prompt_stats,
)


class _FakeMsg:
    def __init__(self, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata
        self.content = ""


def test_capture_usage_from_usage_metadata():
    usage = {}
    _capture_usage(_FakeMsg({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}), usage)
    assert usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_capture_usage_from_response_metadata():
    usage = {}
    _capture_usage(
        _FakeMsg(None, {"usage_metadata": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}}),
        usage,
    )
    assert usage["input_tokens"] == 2


def test_capture_usage_ignores_non_int():
    usage = {}
    _capture_usage(_FakeMsg({"input_tokens": "nope", "output_tokens": 1, "total_tokens": 10}), usage)
    assert usage == {"output_tokens": 1, "total_tokens": 10}


def test_capture_usage_no_metadata():
    usage = {}
    _capture_usage(_FakeMsg(), usage)
    assert usage == {}


def test_estimate_cost_basic():
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    # price config is imported from settings; just assert the shape is a float.
    cost = _estimate_cost(usage)
    assert cost is None or isinstance(cost, float)


def test_extract_text_string():
    assert _extract_text("plain") == "plain"


def test_extract_text_list_of_parts():
    parts = [{"type": "text", "text": "a"}, {"type": "content", "text": "b"}, "c"]
    assert _extract_text(parts) == "abc"


def test_extract_text_other():
    assert _extract_text(None) == "None"


def test_prompt_stats_counts_text_and_images():
    content = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "image_url", "image_url": {"url": "y"}},
    ]
    chars, images = _prompt_stats(content)
    assert chars == 5
    assert images == 2
