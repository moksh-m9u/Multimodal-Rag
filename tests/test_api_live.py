"""Live smoke tests against the real store, watsonx, and Gemini.

Skipped unless selected with ``-m live`` (see ``tests/conftest.py``). They
check status codes and response shapes only, never answer wording.

Running them opens the real Chroma store, which rewrites tracked files under
``dbv2/chroma_db``. Restore them afterwards with
``git checkout -- dbv2/chroma_db``.

Each run makes one Gemini call (``/query`` with ``max_images=1``) and a few
watsonx embedding calls.
"""

from __future__ import annotations

import os

import pytest
from dotenv import find_dotenv, load_dotenv
from fastapi.testclient import TestClient

pytestmark = pytest.mark.live

_REQUIRED_ENV = ("WATSONX_API_KEY", "WATSONX_PROJECT_ID", "GEMINI_API_KEY")
_FIGURE_QUESTION = "Show the pin configuration diagram of the LM555 timer."


@pytest.fixture(scope="module")
def live_client():
    """A TestClient over the real app; fails early, by name, if a key is missing."""
    load_dotenv(find_dotenv(usecwd=True))
    missing = [name for name in _REQUIRED_ENV if not os.getenv(name)]
    if missing:
        pytest.fail(f"missing environment variables: {', '.join(missing)}")

    from api.main import app

    with TestClient(app) as client:
        yield client


def test_health(live_client):
    response = live_client.get("/health")
    assert response.status_code == 200
    # /health answers 200 even when the store fails, so the status field is
    # what proves the store loaded.
    assert response.json()["status"] == "ok"


def test_retrieve_returns_chunks(live_client):
    response = live_client.post("/retrieve", json={"query": _FIGURE_QUESTION, "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["num_chunks"] > 0
    assert all(isinstance(c["enhanced_content"], str) for c in body["chunks"])


def test_query_returns_answer(live_client):
    response = live_client.post(
        "/query", json={"query": _FIGURE_QUESTION, "top_k": 3, "max_images": 1}
    )
    assert response.status_code == 200
    answer = response.json()["answer"]
    assert answer.strip()
    # A provider failure arrives as a 200 with this text, not as an error status.
    assert not answer.startswith("Sorry, could not complete response")


def test_figure_loads(live_client):
    body = live_client.post("/retrieve", json={"query": _FIGURE_QUESTION, "top_k": 10}).json()
    urls = [url for chunk in body["chunks"] for url in chunk["image_urls"]]
    assert urls, "no retrieved chunk had a figure; pick a more visual question"

    response = live_client.get(urls[0])
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"
