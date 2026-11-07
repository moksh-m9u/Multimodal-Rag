"""Tests for the watsonx.ai embedding factory (no network calls)."""

import pytest

import src.embed as embed


def test_builds_watsonx_embeddings_from_settings(monkeypatch):
    captured = {}
    monkeypatch.setattr(embed, "WATSONX_API_KEY", "key")
    monkeypatch.setattr(embed, "WATSONX_PROJECT_ID", "project")
    monkeypatch.setattr(embed, "WatsonxEmbeddings", lambda **kw: captured.update(kw) or "model")

    assert embed.load_embedding_model() == "model"
    assert captured == {
        "model_id": embed.EMBEDDING_MODEL,
        "url": embed.WATSONX_URL,
        "project_id": "project",
        "apikey": "key",
        "params": {"truncate_input_tokens": embed.EMBEDDING_MAX_INPUT_TOKENS},
    }


def test_missing_credentials_are_named(monkeypatch):
    monkeypatch.setattr(embed, "WATSONX_API_KEY", "")
    monkeypatch.setattr(embed, "WATSONX_PROJECT_ID", "")

    with pytest.raises(RuntimeError, match="WATSONX_API_KEY, WATSONX_PROJECT_ID"):
        embed.load_embedding_model()
