"""Tests for the provider/model registry (pure dataclasses, no deps)."""

from config.providers import (
    ALL_PROVIDERS,
    PROVIDER_BY_ID,
    get_model,
    get_provider,
    list_models_for_provider,
)


def test_expected_providers_present():
    ids = {p.id for p in ALL_PROVIDERS}
    assert {"gemini", "groq", "huggingface", "openai"} <= ids


def test_duplicate_provider_ids():
    assert len(PROVIDER_BY_ID) == len(ALL_PROVIDERS)


def test_get_provider_unknown_returns_none():
    assert get_provider("nope") is None


def test_get_model_known():
    model = get_model("gemini", "gemini-3.5-flash-lite")
    assert model is not None
    assert model.context_window > 0


def test_get_model_unknown_returns_none():
    assert get_model("gemini", "does-not-exist") is None
    assert get_model("nope", "x") is None


def test_every_model_has_unique_id():
    seen = set()
    for provider in ALL_PROVIDERS:
        for model in provider.models:
            key = (provider.id, model.id)
            assert key not in seen
            seen.add(key)


def test_list_models_for_provider():
    assert len(list_models_for_provider("gemini")) > 0
