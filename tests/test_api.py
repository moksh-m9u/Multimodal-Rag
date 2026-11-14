"""Offline tests for the HTTP API in ``api/main.py``.

They check each endpoint's status codes and response shape, so a change to
the API layer shows exactly which parts of the HTTP contract it alters.

The real FastAPI app and its lifespan run, but the three heavy things
``RAGService`` would build are replaced before the app starts:

* the watsonx embedding client and the Chroma store (patched in
  ``api.service``, where ``RAGService._load`` looks them up);
* the default Gemini client (``src.retrieval.generate._create_llm``);
* the custom-provider client (``api.custom_service._create_llm_for_model``).

Guards make any attempt to open the real Chroma store or build a real watsonx
client fail the test, and LangSmith tracing is switched off, so no test
touches ``dbv2/`` or the network.
"""

from __future__ import annotations

import base64
import json

import langsmith
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

import api.custom_service
import api.service
import src.embed
import src.retrieval.generate
import src.retrieval.search
from api.main import app

# Starts with the JPEG magic bytes so it looks like a real figure.
JPEG_BYTES = b"\xff\xd8\xff\xe0fake-jpeg-for-tests"


def _fake_chunks() -> list[Document]:
    """Two chunks, one per metadata schema that ``src/documents.py`` reads."""
    nested = Document(
        page_content="LM317 summary: adjustable regulator with ADJ, VOUT, VIN pins.",
        metadata={
            "source": "lm317",
            "original_content": json.dumps(
                {
                    "raw_text": "Pin 1 ADJ, pin 2 VOUT, pin 3 VIN.",
                    "tables_html": ["<table><tr><td>ADJ</td></tr></table>"],
                    "images_base64": [base64.b64encode(JPEG_BYTES).decode()],
                }
            ),
        },
    )
    flat = Document(
        page_content="TLV1117 summary: fixed 3.3 V LDO.",
        metadata={
            "source": "tlv1117",
            "raw_text": "Dropout 1.1 V at 800 mA.",
            "tables_html": "[]",
            "image_paths": "[]",
        },
    )
    return [nested, flat]


class _FakeRetriever:
    """Returns the first ``k`` fake chunks for any question."""

    def __init__(self, k: int):
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        """Mimic a LangChain retriever's ``invoke``."""
        return _fake_chunks()[: self.k]


class _FakeStore:
    """Stands in for the Chroma store: only ``as_retriever`` is used."""

    def as_retriever(self, search_type: str = "mmr", search_kwargs: dict | None = None):
        """Mimic ``Chroma.as_retriever``."""
        return _FakeRetriever((search_kwargs or {}).get("k", 10))


class FakeBackend:
    """Switches the tests flip to make the fake store or LLM fail.

    ``prompts`` records every message list sent to the fake LLM so a test can
    check what the model would have seen.
    """

    def __init__(self):
        self.store_error: str | None = None
        self.llm_error: str | None = None
        self.prompts: list = []

    def load_store(self, embedding_model) -> _FakeStore:
        """Replacement for ``load_vector_store``."""
        if self.store_error:
            raise RuntimeError(self.store_error)
        return _FakeStore()

    def make_llm(self, *args, **kwargs) -> "_FakeLLM":
        """Replacement for both LLM builders; ignores provider, model and key."""
        return _FakeLLM(self)


class _FakeLLM:
    """Streams two tokens, then reports usage on the last chunk like Gemini."""

    def __init__(self, backend: FakeBackend):
        self._backend = backend

    def stream(self, messages):
        """Mimic ``BaseChatModel.stream``."""
        self._backend.prompts.append(messages)
        if self._backend.llm_error:
            raise RuntimeError(self._backend.llm_error)
        yield AIMessageChunk(content="Hello")
        yield AIMessageChunk(
            content=" world",
            usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )


def _forbidden(name: str):
    """Build a stand-in that fails the test if the real ``name`` is constructed."""

    def _raise(*args, **kwargs):
        raise AssertionError(f"offline tests must not construct {name}")

    return _raise


@pytest.fixture
def backend(monkeypatch) -> FakeBackend:
    """Patch every heavy dependency with fakes and turn tracing off."""
    fake = FakeBackend()
    monkeypatch.setattr(api.service, "load_embedding_model", lambda: "fake-embeddings")
    monkeypatch.setattr(api.service, "load_vector_store", fake.load_store)
    monkeypatch.setattr(api.service, "build_retriever", lambda db: db.as_retriever())
    monkeypatch.setattr(src.retrieval.generate, "_create_llm", fake.make_llm)
    monkeypatch.setattr(api.custom_service, "_create_llm_for_model", fake.make_llm)
    monkeypatch.setattr(src.retrieval.search, "Chroma", _forbidden("Chroma"))
    monkeypatch.setattr(src.embed, "WatsonxEmbeddings", _forbidden("WatsonxEmbeddings"))

    # A process-wide switch: it also reaches the threads TestClient and
    # asyncio.to_thread run the app in, which a context manager would not.
    langsmith.configure(enabled=False)
    yield fake
    langsmith.configure(enabled=None)


@pytest.fixture
def client(backend: FakeBackend):
    """A TestClient whose ``with`` block runs the app's lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Split a text/event-stream body into ``(event, data)`` pairs."""
    events = []
    for frame in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_root_lists_endpoints(client):
    body = client.get("/").json()
    assert body["docs"] == "/docs"
    assert "POST /query" in body["endpoints"]


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "vector_store": "loaded"}


def test_health_reports_store_failure(client, backend):
    """A broken store is reported in the body with status 200.

    Known defect: AGENTS.md ("Errors are real HTTP errors") calls for a 503.
    """
    backend.store_error = "store missing"
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "error"
    assert response.json()["detail"] == "store missing"


def test_retrieve_returns_chunks_with_tables_and_images(client):
    response = client.post("/retrieve", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 200
    body = response.json()

    assert body["query"] == "LM317 pins"
    assert body["num_chunks"] == 2
    assert body["total_images"] == 1
    assert body["total_tables"] == 1

    first, second = body["chunks"]
    assert first["chunk_id"] == 1
    assert first["source"] == "lm317"
    assert first["raw_text"] == "Pin 1 ADJ, pin 2 VOUT, pin 3 VIN."
    assert first["has_table"] and first["has_image"]
    assert first["images_base64"] == [base64.b64encode(JPEG_BYTES).decode()]
    assert first["image_urls"] == [f"/images/{body['response_id']}/0"]
    assert "original_content" not in first["metadata"]
    assert second["image_urls"] == [] and not second["has_image"]


def test_retrieve_rejects_bad_input(client):
    assert client.post("/retrieve", json={"query": ""}).status_code == 422
    assert client.post("/retrieve", json={"query": "x", "top_k": 31}).status_code == 422


def test_image_url_serves_jpeg(client):
    body = client.post("/retrieve", json={"query": "LM317 pins", "top_k": 2}).json()
    response = client.get(body["chunks"][0]["image_urls"][0])
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == JPEG_BYTES


def test_unknown_image_is_404(client):
    body = client.post("/retrieve", json={"query": "LM317 pins", "top_k": 2}).json()
    assert client.get(f"/images/{body['response_id']}/5").status_code == 404
    assert client.get("/images/not-a-response-id/0").status_code == 404


def test_query_returns_answer_and_usage(client, backend):
    response = client.post("/query", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 200
    body = response.json()

    assert body["answer"] == "Hello world"
    assert body["answer_characters"] == len("Hello world")
    assert body["num_chunks"] == 2
    assert body["usage"]["input_tokens"] == 10
    assert body["usage"]["prompt_images"] == 1

    content = backend.prompts[0][0].content
    assert "QUESTION: LM317 pins" in content[0]["text"]
    assert content[1]["type"] == "image_url"


def test_query_llm_failure_becomes_answer_text(client, backend):
    """An LLM failure is returned as a 200 whose answer is an apology.

    Known defect: AGENTS.md ("Errors are real HTTP errors") calls for a 502.
    """
    backend.llm_error = "quota exceeded"
    response = client.post("/query", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 200
    assert response.json()["answer"] == "Sorry, could not complete response due to: quota exceeded"


def test_stream_emits_retrieval_tokens_done(client):
    response = client.get("/query/stream", params={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    assert [name for name, _ in events] == ["retrieval", "token", "token", "done"]
    assert events[0][1]["num_chunks"] == 2
    assert [data["delta"] for name, data in events if name == "token"] == ["Hello", " world"]
    done = events[-1][1]
    assert done["answer"] == "Hello world"
    assert done["response_id"] == events[0][1]["response_id"]


def test_custom_query_uses_chosen_model(client):
    response = client.post(
        "/query/custom",
        json={
            "query": "LM317 pins",
            "top_k": 2,
            "provider": "gemini",
            "model": "gemini-3.5-flash-lite",
            "api_key": "test-key",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Hello world"
    assert body["usage"]["model"] == "gemini-3.5-flash-lite"


def test_custom_query_unknown_model_is_500(client):
    """An unknown model is reported as a server error.

    Known defect: it is a client mistake, so AGENTS.md ("Errors are real HTTP
    errors") calls for a 4xx.
    """
    response = client.post(
        "/query/custom",
        json={"query": "x", "provider": "gemini", "model": "no-such-model", "api_key": "k"},
    )
    assert response.status_code == 500
    assert "no-such-model" in response.json()["detail"]
