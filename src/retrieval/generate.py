"""Answer generation using a multimodal LLM.

Builds the prompt from retrieved chunks (enhanced summaries + HTML tables +
base64 images), calls Gemini to generate an answer, and wraps both retrieval
and generation under a single LangSmith trace so the whole query shows up as
one run in the UI.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from langsmith.run_helpers import set_tracing_parent

from config.settings import (
    GENERATION_INPUT_PRICE_PER_1M,
    GENERATION_MODEL,
    GENERATION_OUTPUT_PRICE_PER_1M,
    GENERATION_TEMPERATURE,
)
from src.documents import extract_original_data
from src.logger import get_logger
from src.retrieval.search import retrieve_chunks
from src.tracing import summarize_chunks

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """Answer plus the usage metadata collected during generation."""

    answer: str
    usage: dict[str, Any]


def _capture_usage(obj: Any, usage: dict[str, Any]) -> None:
    """Merge token counts reported by the LLM into ``usage``.

    ``obj`` is the AIMessage/AIMessageChunk returned by the provider.  Usage
    may arrive on ``usage_metadata`` or nested in ``response_metadata``; for
    streams it is typically present on every chunk (cumulative), so later
    chunks overwrite earlier ones.
    """
    raw = getattr(obj, "usage_metadata", None)
    if not raw:
        raw = (getattr(obj, "response_metadata", None) or {}).get("usage_metadata")
    if not raw:
        return
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        if key in raw and isinstance(raw[key], int):
            usage[key] = raw[key]


def _estimate_cost(usage: dict[str, Any]) -> float | None:
    """Estimate generation cost (USD) from token usage and configured pricing."""
    if not GENERATION_INPUT_PRICE_PER_1M and not GENERATION_OUTPUT_PRICE_PER_1M:
        return None
    inp = usage.get("input_tokens") or 0
    out = usage.get("output_tokens") or 0
    cost = (
        inp / 1_000_000 * GENERATION_INPUT_PRICE_PER_1M
        + out / 1_000_000 * GENERATION_OUTPUT_PRICE_PER_1M
    )
    return round(cost, 6)


def _prompt_stats(message_content: list[dict]) -> tuple[int, int]:
    """Return ``(text_characters, image_count)`` of a message content list."""
    text = next((x["text"] for x in message_content if x["type"] == "text"), "")
    images = sum(1 for x in message_content if x["type"] == "image_url")
    return len(text), images


def _build_text_prompt(chunks: list[Document], query: str) -> str:
    """Build the text portion of the prompt from retrieved chunks.

    Sends enhanced summaries as primary context, with HTML tables appended.
    Images are sent separately via :func:`_collect_images`.

    Note: only the first 5 chunks are included to keep the prompt size
    predictable for the model.
    """
    parts = [
        "You are given retrieved chunks from a technical document.\n"
        "Each chunk contains:\n"
        "- a searchable summary of the original content,\n"
        "- optional HTML tables,\n"
        "- optional attached figures/images.\n\n"
        "Treat the summaries as authoritative representations of the document.\n"
        "Do NOT claim information is absent unless you have examined ALL provided chunks.\n"
        "If a chunk explicitly contains the answer, answer directly using that chunk.\n"
        "Use both the text summaries and the attached images when answering.\n",
        f"QUESTION: {query}\n",
        "RETRIEVED CONTEXT:",
        "",
    ]

    for i, chunk in enumerate(chunks[:5]):
        enhanced = chunk.page_content
        if enhanced:
            parts.append(f"--- Chunk {i + 1} ---")
            parts.append(f"SUMMARY:\n{enhanced.strip()}\n")

        tables_html = extract_original_data(chunk)["tables_html"]
        if tables_html:
            parts.append("TABLES:")
            for j, table in enumerate(tables_html):
                parts.append(f"Table {j + 1}:\n{table}\n")

        parts.append("")

    parts.append(
        "The images above are figures from the document. "
        "Use the summaries, tables, and images together to provide a detailed answer."
    )

    return "\n".join(parts)


def _collect_images(chunks: list[Document]) -> list[dict]:
    """Collect all base64 images from chunks into message content blocks."""
    image_blocks: list[dict] = []
    for chunk in chunks:
        for b64 in extract_original_data(chunk)["images_base64"]:
            image_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
    return image_blocks


def _build_message_content(chunks: list[Document], query: str) -> list[dict]:
    """Build the full multimodal message content list (text + images)."""
    content: list[dict] = [{"type": "text", "text": _build_text_prompt(chunks, query)}]
    content.extend(_collect_images(chunks))
    return content


def _create_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini LLM instance used for generation."""
    logger.info("  Using model: %s", GENERATION_MODEL)
    return ChatGoogleGenerativeAI(
        model=GENERATION_MODEL,
        temperature=GENERATION_TEMPERATURE,
    )


def _save_prompt_debug(message_content: list[dict]) -> None:
    """Dump the full text prompt to ``last_prompt.txt`` for inspection."""
    from pathlib import Path

    text_content = next(x["text"] for x in message_content if x["type"] == "text")
    image_count = sum(1 for x in message_content if x["type"] == "image_url")
    Path("last_prompt.txt").write_text(text_content, encoding="utf-8")
    logger.info(
        "Prompt saved to last_prompt.txt (%d chars, %d images)",
        len(text_content),
        image_count,
    )


@traceable(run_type="llm", name="GenerateAnswer")
def generate_answer(
    chunks: list[Document], query: str, verbose: bool = False
) -> GenerationResult:
    """Generate a final answer using the multimodal LLM.

    Args:
        chunks: Retrieved document chunks.
        query: The original user query.
        verbose: If True, logs additional debug detail.

    Returns:
        A :class:`GenerationResult` with the answer string and usage metadata
        (tokens, model, latency, prompt size, estimated cost).
    """
    try:
        chunk_summaries = summarize_chunks(chunks)
        total_images = sum(c.get("image_count", 0) for c in chunk_summaries)
        total_tables = sum(c.get("table_count", 0) for c in chunk_summaries)

        llm = _create_llm()
        message_content = _build_message_content(chunks, query)
        _save_prompt_debug(message_content)

        message = HumanMessage(content=message_content)
        start = time.perf_counter()
        response = llm.invoke([message])
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        prompt_chars, prompt_images = _prompt_stats(message_content)
        usage: dict[str, Any] = {
            "model": GENERATION_MODEL,
            "latency_ms": latency_ms,
            "prompt_chars": prompt_chars,
            "prompt_images": prompt_images,
            "chunks_used": min(len(chunks), 5),
        }
        _capture_usage(response, usage)
        usage["estimated_cost_usd"] = _estimate_cost(usage)

        logger.info(
            "LangSmith trace: %d chunks, %d images, %d tables; usage=%s",
            len(chunks),
            total_images,
            total_tables,
            usage,
        )
        return GenerationResult(answer=response.content, usage=usage)

    except Exception as e:
        error_msg = f"Answer generation failed: {e}"
        if verbose:
            logger.exception(error_msg)
        else:
            logger.warning(error_msg)
        return GenerationResult(answer=error_msg, usage={"error": error_msg})


def _collect_image_attachments(chunks: list[Document]) -> dict:
    """Gather images from chunks as binary attachments for LangSmith traces.

    Uses :func:`extract_original_data` to resolve either storage schema
    (file-path or inline base64) to image bytes.
    """
    attachments: dict = {}
    img_idx = 0
    for chunk in chunks:
        for b64_str in extract_original_data(chunk)["images_base64"]:
            try:
                attachments[f"chunk_img_{img_idx}"] = ("image/jpeg", base64.b64decode(b64_str))
                img_idx += 1
            except Exception:
                logger.warning("Could not decode image attachment for trace")
    return attachments


def _attach_images_to_trace(run_tree, chunks: list[Document]) -> None:
    """Attach retrieved chunk images to ``run_tree`` if any are available."""
    if run_tree is None:
        return
    attachments = _collect_image_attachments(chunks)
    if attachments:
        run_tree.attachments = attachments


class _StreamWrapper:
    """Iterator adapter that keeps the LangSmith parent context alive.

    ``generate_answer_stream`` is a generator, so LangSmith defers creating
    its trace run until the first ``next()`` call.  By the time that happens
    the original parent context (inside ``answer_query_stream``) is gone, so
    we re-activate it around every ``next()`` to keep retrieval + generation
    nested under the same ``AnswerQuery`` root run.

    ``usage`` is a shared mutable dict that the streaming generator fills in
    as it runs, so it is populated once the stream is fully consumed.
    """

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


@traceable(run_type="chain", name="AnswerQuery", dangerously_allow_filesystem=True)
def answer_query(retriever, query: str, run_tree=None) -> str:
    """Retrieve chunks then generate an answer under a single LangSmith trace.

    Images from retrieved chunks are attached to the trace so they render in
    the LangSmith UI.  ``run_tree`` is injected by LangSmith.

    Args:
        retriever: Vector store retriever.
        query: The user's question.

    Returns:
        The generated answer string.
    """
    chunks = retrieve_chunks(retriever, query)
    _attach_images_to_trace(run_tree, chunks)
    return generate_answer(chunks, query).answer


@traceable(run_type="chain", name="AnswerQuery", dangerously_allow_filesystem=True)
def answer_query_stream(retriever, query: str, run_tree=None):
    """Retrieve chunks then stream an answer under a single LangSmith trace.

    Yields answer tokens as they are produced.  The retrieved chunks are
    available via the returned generator's ``chunks`` attribute once
    retrieval completes.  ``run_tree`` is injected by LangSmith.

    Args:
        retriever: Vector store retriever.
        query: The user's question.

    Yields:
        Answer token strings.  The wrapper's ``chunks`` and ``usage``
        attributes expose the retrieved chunks and token usage once done.
    """
    chunks = retrieve_chunks(retriever, query)
    _attach_images_to_trace(run_tree, chunks)

    usage: dict[str, Any] = {}
    return _StreamWrapper(
        generate_answer_stream(chunks, query, usage=usage),
        chunks,
        run_tree,
        usage=usage,
    )


@traceable(run_type="llm", name="GenerateAnswerStream")
def generate_answer_stream(
    chunks: list[Document],
    query: str,
    verbose: bool = False,
    usage: dict[str, Any] | None = None,
):
    """Generate a streaming answer using the multimodal LLM.

    Yields answer tokens as they are generated by the LLM.  ``usage`` is an
    optional shared dict (populated in-place as the stream runs) used by
    :func:`answer_query_stream` to expose token usage to callers; a fresh one
    is created when omitted.

    Args:
        chunks: Retrieved document chunks.
        query: The original user query.
        verbose: If True, logs additional debug detail.
        usage: Optional mutable dict to receive usage metadata.

    Yields:
        Answer token strings.
    """
    if usage is None:
        usage = {}

    def _generate():
        try:
            chunk_summaries = summarize_chunks(chunks)
            total_images = sum(c.get("image_count", 0) for c in chunk_summaries)
            total_tables = sum(c.get("table_count", 0) for c in chunk_summaries)

            llm = _create_llm()

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

            message_content = _build_message_content(chunks, query)
            _save_prompt_debug(message_content)

            prompt_chars, prompt_images = _prompt_stats(message_content)
            usage.update(
                {
                    "model": GENERATION_MODEL,
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
                if chunk.content:
                    yield chunk.content
                _capture_usage(chunk, usage)
            usage["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
            usage["estimated_cost_usd"] = _estimate_cost(usage)

        except Exception as e:
            logger.warning("Answer generation failed: %s", e)
            usage["error"] = str(e)
            yield f"Sorry, could not complete response due to: {e}"

    return _generate()
