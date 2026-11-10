"""Tests for the custom-provider query path (no network or vector store)."""

import asyncio

import pytest

import api.custom_service as custom


class _FakeStream:
    """Stands in for the traced answer stream: yields tokens, exposes chunks."""

    def __init__(self):
        self.chunks = ["chunk"]
        self.usage = {"model": "m"}

    def __iter__(self):
        return iter(["Hello", " world"])


class _FakeResponse:
    def model_dump(self):
        return {"response_id": "shared"}


class _FakeService:
    """Records calls so the test can assert the shared service is reused."""

    def __init__(self):
        self.calls = []

    async def ensure_ready(self):
        self.calls.append("ensure_ready")

    def get_retriever(self, top_k):
        self.calls.append(("get_retriever", top_k))
        return "retriever"

    def to_retrieve_response(self, query, chunks):
        self.calls.append(("to_retrieve_response", query, chunks))
        return _FakeResponse()


def test_custom_query_reuses_shared_service(monkeypatch):
    monkeypatch.setattr(custom, "_create_llm_for_model", lambda *a, **k: "llm")
    monkeypatch.setattr(custom, "answer_query_stream_custom", lambda *a, **k: _FakeStream())
    service = _FakeService()

    answer, base, usage = asyncio.run(
        custom.run_custom_query(service, "q", 7, "gemini", "gemini-3.5-flash-lite", "key")
    )

    assert answer == "Hello world"
    assert base == {"response_id": "shared"}
    assert usage == {"model": "m"}
    assert service.calls == [
        "ensure_ready",
        ("get_retriever", 7),
        ("to_retrieve_response", "q", ["chunk"]),
    ]


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="not found"):
        asyncio.run(custom.run_custom_query(_FakeService(), "q", 5, "gemini", "nope", "key"))
