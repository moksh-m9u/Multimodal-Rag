"""Custom LLM query service supporting multiple providers.

Accepts provider, model, and API key at request time, constructs the
appropriate LangChain chat model, and streams the answer using the
same retrieval + prompt logic as the default pipeline.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langsmith import traceable
from langsmith.run_helpers import set_tracing_parent

from config.providers import get_model
from src.documents import extract_original_data
from src.logger import get_logger
from src.retrieval.generate import (
    _build_message_content,
    _capture_usage,
    _estimate_cost,
    _prompt_stats,
    summarize_chunks,
)
from src.retrieval.search import retrieve_chunks

if TYPE_CHECKING:
    from api.service import RAGService

logger = get_logger(__name__)


def _create_llm_for_model(provider_id: str, model_id: str, api_key: str, temperature: float = 0.0, max_tokens: int = 512, thinking: bool = False):
    """Create a LangChain chat model instance for the given provider/model."""

    if model_id.startswith("gemini") or model_id.startswith("models/gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_id,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=api_key,
        )
    elif model_id.startswith("gpt-") or model_id.startswith("o1") or model_id.startswith("o3") or model_id.startswith("gpt-5"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    else:
        from langchain_openai import ChatOpenAI
        if provider_id == "huggingface":
            base_url = "https://router.huggingface.co/v1"
        else:
            base_url = "https://api.groq.com/openai/v1"

        extra: dict = {}
        # Groq supports reasoning_effort to control thinking natively
        if provider_id == "groq":
            extra["reasoning_effort"] = "default" if thinking else "none"

        return ChatOpenAI(
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            model_kwargs=extra,
        )


def _extract_text(content: Any) -> str:
    """Extract plain text from LLM response content (string or list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


class _StreamWrapper:
    """Iterator adapter that keeps the LangSmith parent context alive."""
    def __init__(self, gen, chunks: list[Document], parent=None, usage: dict | None = None):
        self._gen = gen
        self.chunks = chunks
        self._parent = parent
        self.usage: dict[str, Any] = usage if usage is not None else {}

    def __iter__(self):
        return self

    def __next__(self):
        if self._parent is not None:
            with set_tracing_parent(self._parent):
                return next(self._gen)
        return next(self._gen)


@traceable(run_type="llm", name="GenerateAnswerCustom")
def generate_answer_stream(
    chunks: list[Document],
    query: str,
    llm,
    model_id: str,
    verbose: bool = False,
    usage: dict[str, Any] | None = None,
    max_images: int = 0,
    thinking: bool = False,
):
    """Generate a streaming answer using a custom LLM."""
    if usage is None:
        usage = {}

    def _generate():
        try:
            chunk_summaries = summarize_chunks(chunks)
            total_images = sum(c.get("image_count", 0) for c in chunk_summaries)
            total_tables = sum(c.get("table_count", 0) for c in chunk_summaries)

            logger.info("  %d chunks received by generation", len(chunks))
            for ci, c in enumerate(chunks[:5]):
                od = extract_original_data(c)
                logger.debug(
                    "    Chunk %d: summary=%d chars, tables=%d, images=%d",
                    ci + 1,
                    len(c.page_content or ""),
                    len(od["tables_html"]),
                    len(od["images_base64"]),
                )

            message_content = _build_message_content(chunks, query, max_images)
            if thinking:
                # Prepend thinking instruction to the text prompt
                thinking_instruction = (
                    "\n\nBefore answering, show your reasoning step-by-step "
                    "wrapped in <think> and <think> tags. "
                    "Then provide your final answer after the </think> tag."
                )
                message_content[0]["text"] += thinking_instruction

            prompt_chars, prompt_images = _prompt_stats(message_content)
            usage.update(
                {
                    "model": model_id,
                    "prompt_chars": prompt_chars,
                    "prompt_images": prompt_images,
                    "chunks_used": min(len(chunks), 5),
                }
            )

            logger.info(
                "LangSmith trace: %d chunks, %d images, %d tables",
                len(chunks),
                total_images,
                total_tables,
            )

            message = HumanMessage(content=message_content)
            start = time.perf_counter()
            for chunk in llm.stream([message]):
                token_text = _extract_text(chunk.content)
                if token_text:
                    yield token_text
                _capture_usage(chunk, usage)
            usage["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
            usage["estimated_cost_usd"] = _estimate_cost(usage)

        except Exception as e:
            logger.warning("Answer generation failed: %s", e)
            usage["error"] = str(e)
            yield f"Sorry, could not complete response due to: {e}"

    return _generate()


@traceable(run_type="chain", name="AnswerQueryCustom", dangerously_allow_filesystem=True)
def answer_query_stream_custom(
    retriever,
    query: str,
    llm,
    model_id: str,
    run_tree=None,
    max_images: int = 0,
    thinking: bool = False,
):
    """Retrieve chunks then stream an answer with a custom LLM."""
    chunks = retrieve_chunks(retriever, query)
    # Attach images to trace
    from src.retrieval.generate import _attach_images_to_trace
    _attach_images_to_trace(run_tree, chunks)

    usage: dict[str, Any] = {}
    return _StreamWrapper(
        generate_answer_stream(chunks, query, llm, model_id, usage=usage, max_images=max_images, thinking=thinking),
        chunks,
        run_tree,
        usage=usage,
    )


async def run_custom_query(
    service: "RAGService",
    query: str,
    top_k: int,
    provider_id: str,
    model_id: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_images: int = 0,
    thinking: bool = False,
) -> tuple[str, dict, dict]:
    """Run a complete custom query: retrieve, then generate with the chosen model.

    Reuses the application's :class:`RAGService` so the vector store is loaded
    once per process and figures are registered in the shared image registry
    that backs ``GET /images/{response_id}/{index}``.

    Returns:
        A tuple of ``(answer, retrieve_response_dict, usage_dict)``.

    Raises:
        ValueError: If ``model_id`` is not registered for ``provider_id``.
    """
    if not get_model(provider_id, model_id):
        raise ValueError(f"Model {model_id} not found for provider {provider_id}")

    await service.ensure_ready()
    retriever = service.get_retriever(top_k)
    llm = _create_llm_for_model(provider_id, model_id, api_key, temperature, max_tokens, thinking)

    def _run() -> tuple[str, list, dict]:
        stream = answer_query_stream_custom(retriever, query, llm, model_id, max_images=max_images, thinking=thinking)
        answer = "".join(stream)
        return answer, stream.chunks, stream.usage

    answer, chunks, usage = await asyncio.to_thread(_run)
    return answer, service.to_retrieve_response(query, chunks).model_dump(), usage
