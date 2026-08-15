"""Tools available to the orchestrator agent.

Three information-gathering tools plus one planning tool:

* ``retrieve_knowledge_base`` -- searches the project's Chroma knowledge base.
* ``web_search`` -- Tavily live web search, used when the KB lacks relevant info.
* ``extract_urls`` -- Tavily page extraction for deep-diving into sources.
* ``update_todos`` -- the planning tool; records the agent's plan of action.

Each ``@tool`` has full type hints and a rich docstring so the LLM sees clear
descriptions (and the function signatures are reflected into the tool schema).
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Callable

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from tavily import TavilyClient

from config.settings import (
    ORCHESTRATOR_KB_TOP_K,
    ORCHESTRATOR_MAX_EXTRACT_URLS,
    ORCHESTRATOR_SEARCH_RESULTS,
    TAVILY_API_KEY,
)
from src.logger import get_logger
from src.orchestrator.state import OrchestratorState

logger = get_logger(__name__)

_URL_PATTERN = re.compile(r"https?://[^\s\)\]\}>,;\"']+")

_tavily_client: TavilyClient | None = None


def _get_tavily_client() -> TavilyClient:
    """Return a lazily-created Tavily client (keyless mode when no key set)."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = (
            TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else TavilyClient()
        )
    return _tavily_client


# ---------------------------------------------------------------------------
# Knowledge base tool
# ---------------------------------------------------------------------------


def build_kb_tool(retriever_provider: Callable[[], Any] | None = None):
    """Build the knowledge-base search tool.

    Args:
        retriever_provider: Callable returning a Chroma retriever. When the
            orchestrator runs inside the API service this points at the
            already-loaded vector store so it is not loaded twice. When None,
            the tool lazily loads its own store (for standalone scripts).

    Returns:
        A ``@tool``-decorated ``retrieve_knowledge_base`` function.
    """
    if retriever_provider is None:

        def _default_provider():
            from src.embed import load_embedding_model
            from src.retrieval.search import build_retriever, load_vector_store

            return build_retriever(load_vector_store(load_embedding_model()))

        provider = _default_provider
    else:
        provider = retriever_provider

    @tool
    def retrieve_knowledge_base(query: str, top_k: int = ORCHESTRATOR_KB_TOP_K) -> str:
        """Search the internal knowledge base for chunks relevant to a query.

        Use this FIRST for every question. The knowledge base contains
        datasheets, application notes and technical documents for components
        such as voltage regulators. Returns enhanced summaries of the top
        matching chunks, or an explicit message when nothing relevant is found
        (in which case fall back to web_search).

        Args:
            query: The aspect of the question to search for (include exact part
                numbers, e.g. "LM2596"). Make one focused call per component.
            top_k: Maximum number of chunks to return (default 5).

        Returns:
            Formatted chunk summaries, or a note that nothing was found.
        """
        logger.info("KB tool: query=%r top_k=%s", query, top_k)
        retriever = provider()
        try:
            chunks = retriever.invoke(query)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("KB retrieval failed: %s", exc)
            return "Knowledge base search failed. Use web_search instead."

        if not chunks:
            return "No relevant chunks found in the knowledge base. Use web_search."

        parts = [f"Found {len(chunks)} relevant chunk(s) in the knowledge base:"]
        for i, chunk in enumerate(chunks, start=1):
            content = (chunk.page_content or "").strip()
            source = (chunk.metadata or {}).get("source", "unknown")
            parts.append(f"--- KB chunk {i} (source: {source}) ---")
            parts.append(content[:1500])
            parts.append("")
        return "\n".join(parts)

    return retrieve_knowledge_base


# ---------------------------------------------------------------------------
# Web tools
# ---------------------------------------------------------------------------


@tool
def web_search(query: str, max_results: int = ORCHESTRATOR_SEARCH_RESULTS) -> str:
    """Search the live web with Tavily and return the top results.

    Use when the knowledge base has nothing relevant, when you need current
    information (availability, pricing, newer alternatives), or to
    cross-check claims. Returns each result's title, URL, a content snippet
    and a relevance score.

    Args:
        query: A focused web query (include exact part numbers and keywords).
        max_results: Maximum number of results to return (default 5).

    Returns:
        Formatted search results.
    """
    logger.info("web_search: query=%r max_results=%s", query, max_results)
    try:
        response = _get_tavily_client().search(
            query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Web search failed: %s", exc)
        return f"Web search failed: {exc}"

    parts: list[str] = []
    answer = response.get("answer")
    if answer:
        parts.append(f"Tavily summary: {answer}")
        parts.append("")

    results = response.get("results", [])
    if not results:
        return "Web search returned no results. Try different keywords."

    parts.append("URLS:")
    parts.extend(f"- {res.get('url', '')}" for res in results)
    parts.append("")

    for i, res in enumerate(results, start=1):
        url = res.get("url", "")
        title = res.get("title") or url
        raw = (res.get("raw_content") or res.get("content") or "")[:800]
        score = res.get("score")
        parts.append(f"[{i}] {title}")
        parts.append(f"    URL: {url}")
        if score is not None:
            parts.append(f"    Relevance: {score:.3f}")
        if raw:
            parts.append(f"    Content: {raw}")
        parts.append("")
    return "\n".join(parts).strip()


@tool
def extract_urls(urls: list[str], query: str = "") -> str:
    """Read the full content of specific web pages.

    Use to deep-dive into the most promising ``web_search`` results before
    citing them. Only call this on URLs you have actually seen in search
    results. At most the first few URLs are read (the rest are dropped).

    Args:
        urls: The URLs to extract content from (max 4, earliest kept).
        query: Optional hint to focus extraction (e.g. the part number).

    Returns:
        Formatted raw content per successfully extracted URL.
    """
    clean = list(dict.fromkeys(urls))[:ORCHESTRATOR_MAX_EXTRACT_URLS]
    logger.info("extract_urls: urls=%s query=%r", clean, query)
    try:
        response = _get_tavily_client().extract(
            urls=clean,
            extract_depth="advanced",
            query=query or None,
            chunks_per_source=3,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("URL extraction failed: %s", exc)
        return f"URL extraction failed: {exc}"

    parts: list[str] = []
    parts.append("URLS:")
    parts.extend(f"- {url}" for url in clean)
    parts.append("")
    for res in response.get("results", []):
        url = res.get("url", "")
        content = res.get("raw_content") or res.get("content") or ""
        parts.append(f"--- Extracted: {url} ---")
        parts.append(content[:4000])
        parts.append("")
    for failed in response.get("failed_results", []):
        parts.append(f"[failed to extract] {failed.get('url', '?')}: {failed.get('error', '?')}")

    if not parts:
        return "URL extraction returned no readable content."
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Planning tool
# ---------------------------------------------------------------------------


@tool
def update_todos(
    todos: list[str], state: Annotated[OrchestratorState, InjectedState]
) -> str:
    """Record the plan of steps you will take to answer the user's question.

    Call this at the start of every response, before doing any research, to
    write down 3-6 concrete steps. The steps are shown to the user so they can
    follow your reasoning.

    Args:
        todos: The list of planned steps, in order.
        state: Conversation state (injected automatically).

    Returns:
        A short confirmation message.
    """
    logger.info("update_todos: %s", todos)
    state["todos"] = todos
    return f"Plan recorded ({len(todos)} steps)."


# ---------------------------------------------------------------------------
# Helpers shared with the graph layer
# ---------------------------------------------------------------------------


def extract_urls_from_text(text: str) -> list[str]:
    """Extract unique URLs from a string (used to build the sources list)."""
    return list(dict.fromkeys(_URL_PATTERN.findall(text or "")))


def _urls_block(text: str) -> str:
    """Return just the lines of the ``URLS:`` block (stops at the first blank line)."""
    if "URLS:" not in text:
        return ""
    after = text.split("URLS:", 1)[1]
    lines: list[str] = []
    started = False
    for line in after.splitlines():
        if not line.strip():
            if started:
                break
            continue
        started = True
        lines.append(line)
    return "\n".join(lines)


def extract_urls_from_messages(messages: list) -> list[str]:
    """Collect URLs from the ``URLS:`` block of web tool results only."""
    urls: list[str] = []
    for message in messages:
        name = getattr(message, "name", None)
        if name not in ("web_search", "extract_urls"):
            continue
        content = getattr(message, "content", "") or ""
        urls.extend(extract_urls_from_text(_urls_block(content)))
    return list(dict.fromkeys(urls))
