"""Graph construction, LLM factory and run/stream helpers for the orchestrator.

Graph layout::

    START -> agent -> tools -> plan_sink -> agent -> ... -> END

* ``agent`` -- the LLM, bound to the tool set, with the system prompt prepended.
* ``tools`` -- executes any tool calls the agent made.
* ``plan_sink`` -- extracts the plan (``update_todos``) and sources (web
  results) from the message history into state channels for the UI.

The graph is compiled with an in-memory checkpointer, so conversation memory
is preserved per ``thread_id`` (pass one in the config to continue a thread).
"""

from __future__ import annotations

import asyncio
import queue
import re
import threading
import uuid
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from config.settings import (
    GROQ_API_KEY,
    ORCHESTRATOR_FALLBACK,
    ORCHESTRATOR_GROQ_MODEL,
    ORCHESTRATOR_MODEL,
    ORCHESTRATOR_PROVIDER,
    ORCHESTRATOR_RECURSION_LIMIT,
    ORCHESTRATOR_TEMPERATURE,
)
from src.logger import get_logger
from src.orchestrator.prompt import SYSTEM_PROMPT
from src.orchestrator.state import OrchestratorState
from src.orchestrator.tools import (
    build_kb_tool,
    extract_urls,
    extract_urls_from_messages,
    update_todos,
    web_search,
)

logger = get_logger(__name__)

_TOOL_NAMES = {"retrieve_knowledge_base", "web_search", "extract_urls", "update_todos"}


class _FallbackChatModel(BaseChatModel):
    """A chat model that transparently retries with a fallback provider.

    The primary model (e.g. Gemini) is tried first; if it raises, the whole
    call is replayed against the fallback (e.g. Groq) so the agent keeps
    running. ``bind_tools`` forwards the tools to both providers.

    Streaming tokens are re-emitted through LangChain's run manager, so
    ``astream_events`` still yields ``on_chat_model_stream`` events (the graph
    streams token deltas to the UI). ``used_label`` reports which model
    actually answered, so the API can surface it in responses.
    """

    @property
    def _llm_type(self) -> str:
        return "fallback_chat_model"

    primary: BaseChatModel | None = None
    fallback: BaseChatModel | None = None

    def __init__(
        self,
        primary: BaseChatModel,
        fallback: BaseChatModel,
        primary_label: str,
        fallback_label: str,
    ) -> None:
        super().__init__()
        self.primary = primary
        self.fallback = fallback
        self._primary_label = primary_label
        self._fallback_label = fallback_label
        self._last_used: str | None = None

    @property
    def model(self) -> str:
        return self._primary_label

    @property
    def used_label(self) -> str:
        return self._last_used or self._primary_label

    def bind_tools(self, tools: list, *, tool_choice=None, **kwargs):
        self.primary = self.primary.bind_tools(tools, tool_choice=tool_choice, **kwargs)
        self.fallback = self.fallback.bind_tools(tools, tool_choice=tool_choice, **kwargs)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last_error: Exception | None = None
        for label, model in (
            (self._primary_label, self.primary),
            (self._fallback_label, self.fallback),
        ):
            try:
                merged = None
                for chunk in model.stream(messages, **kwargs):
                    msg_chunk = chunk.message if isinstance(chunk, ChatGenerationChunk) else chunk
                    if merged is None:
                        merged = chunk
                    else:
                        merged += chunk
                    if run_manager:
                        token = _content_to_text(msg_chunk.content)
                        if token:
                            run_manager.on_llm_new_token(token)
                if merged is None:
                    raise RuntimeError("Model produced no output")
                self._last_used = label
                final = merged.message if isinstance(merged, ChatGenerationChunk) else merged
                message = AIMessage(
                    content=final.content,
                    tool_calls=final.tool_calls or [],
                    additional_kwargs=final.additional_kwargs,
                )
                return ChatResult(generations=[ChatGeneration(message=message)])
            except Exception as exc:  # noqa: BLE001 - fall through to backup
                last_error = exc
                logger.warning(
                    "Orchestrator LLM %s failed (%s); retrying with fallback %s",
                    label,
                    str(exc)[:200],
                    self._fallback_label,
                )
        raise last_error  # type: ignore[misc]


def create_orchestrator_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """Create the reasoning LLM for the orchestrator.

    Args:
        provider: ``"gemini"`` (default) or ``"groq"``. Reads the API key from
            the environment via the LangChain integration classes.
        model: Model name override (defaults to ``ORCHESTRATOR_MODEL``).
        temperature: Sampling temperature (defaults to ``ORCHESTRATOR_TEMPERATURE``).

    Returns:
        A chat model instance bound to no tools (tools are bound by the graph).
        When ``provider`` is ``"gemini"`` and ``ORCHESTRATOR_FALLBACK`` is
        enabled (and a Groq key exists), the result is a
        :class:`_FallbackChatModel` that retries with ``ORCHESTRATOR_GROQ_MODEL``
        if the primary call fails.
    """
    provider = (provider or ORCHESTRATOR_PROVIDER).lower()
    if model is None:
        model = ORCHESTRATOR_GROQ_MODEL if provider == "groq" else ORCHESTRATOR_MODEL
    temperature = ORCHESTRATOR_TEMPERATURE if temperature is None else temperature
    logger.info("Orchestrator LLM: provider=%s model=%s temperature=%s", provider, model, temperature)

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature)

    from langchain_google_genai import ChatGoogleGenerativeAI

    primary = ChatGoogleGenerativeAI(model=model, temperature=temperature)
    if not (ORCHESTRATOR_FALLBACK and GROQ_API_KEY):
        logger.info("Groq fallback disabled (ORCHESTRATOR_FALLBACK=%s, key set=%s)", ORCHESTRATOR_FALLBACK, bool(GROQ_API_KEY))
        return primary

    from langchain_groq import ChatGroq

    fallback = ChatGroq(model=ORCHESTRATOR_GROQ_MODEL, temperature=temperature)
    logger.info("Wrapping primary %s with Groq fallback %s", model, ORCHESTRATOR_GROQ_MODEL)
    return _FallbackChatModel(
        primary,
        fallback,
        primary_label=model,
        fallback_label=ORCHESTRATOR_GROQ_MODEL,
    )


def build_kb_tool_for_retriever(retriever_provider: Callable[[], Any]):
    """Convenience wrapper keeping the KB tool construction explicit."""
    return build_kb_tool(retriever_provider)


def _build_tools(retriever_provider: Callable[[], Any] | None = None) -> list:
    """Assemble the full tool set for the agent."""
    return [
        update_todos,
        build_kb_tool(retriever_provider),
        web_search,
        extract_urls,
    ]


def _plan_sink(state: OrchestratorState) -> dict:
    """Fold plan + sources from the message history into state channels."""
    todos: list[str] = []
    kb_searches = 0
    kb_chunks = 0
    web_urls: list[str] = []

    for message in state["messages"]:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls or []:
                if tool_call["name"] == "update_todos":
                    todos = list(tool_call["args"].get("todos", []))
        name = getattr(message, "name", None)
        if name == "retrieve_knowledge_base":
            kb_searches += 1
            match = re.search(r"Found (\d+) relevant chunk", getattr(message, "content", "") or "")
            if match:
                kb_chunks += int(match.group(1))

    web_urls = extract_urls_from_messages(state["messages"])

    sources: list[dict] = [{
        "type": "kb",
        "label": f"{kb_searches} KB search(es), {kb_chunks} chunk(s) grounded",
    }]
    sources.extend({"type": "web", "url": url} for url in dict.fromkeys(web_urls))

    return {"todos": todos, "sources": sources}


def build_orchestrator(
    retriever_provider: Callable[[], Any] | None = None,
    llm: BaseChatModel | None = None,
    checkpointer: Any | None = None,
):
    """Build and compile the orchestrator graph.

    Args:
        retriever_provider: Callable returning the Chroma retriever the
            knowledge-base tool should use. Pass ``None`` (or omit) to let the
            tool lazily load its own store — handy for standalone scripts.
        llm: Optional pre-built chat model; otherwise created via
            :func:`create_orchestrator_llm` from settings.
        checkpointer: Optional checkpointer shared across graphs (keeps thread
            memory alive when the model changes). Defaults to a fresh
            :class:`MemorySaver`.

    Returns:
        A compiled ``StateGraph`` with an in-memory checkpointer.
    """
    tools = _build_tools(retriever_provider)
    if llm is None:
        llm = create_orchestrator_llm()
    model = llm.bind_tools(tools)

    def agent(state: OrchestratorState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [model.invoke(messages)]}

    def route(state: OrchestratorState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(OrchestratorState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("plan_sink", _plan_sink)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "plan_sink")
    graph.add_edge("plan_sink", "agent")
    return graph.compile(checkpointer=checkpointer or MemorySaver())


# ---------------------------------------------------------------------------
# Run / stream helpers
# ---------------------------------------------------------------------------


def _make_config(thread_id: str | None) -> dict:
    """Build a LangGraph config, generating a thread_id when none is given."""
    if thread_id is None:
        thread_id = uuid.uuid4().hex
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": ORCHESTRATOR_RECURSION_LIMIT,
    }


def _build_result(state: dict, thread_id: str) -> dict:
    """Extract the final answer, plan, sources and tools from graph state."""
    messages = state.get("messages", [])
    answer = ""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            answer = _content_to_text(message.content)
            break
    tools_used = sorted(
        {
            call["name"]
            for message in messages
            if isinstance(message, AIMessage)
            for call in (message.tool_calls or [])
        }
    )
    return {
        "answer": answer,
        "thread_id": thread_id,
        "todos": state.get("todos", []),
        "sources": state.get("sources", []),
        "tools_used": tools_used,
    }


def run_orchestrator(
    graph,
    query: str,
    thread_id: str | None = None,
    config: dict | None = None,
) -> dict:
    """Run the orchestrator synchronously and return the final result dict.

    Args:
        graph: A compiled orchestrator graph (from :func:`build_orchestrator`).
        query: The user's question.
        thread_id: Conversation id for memory (auto-generated when ``None``).
        config: Optional override config.

    Returns:
        Dict with ``answer``, ``thread_id``, ``todos``, ``sources`` and
        ``tools_used`` keys.
    """
    cfg = config or _make_config(thread_id)
    thread_id = cfg["configurable"]["thread_id"]
    input_ = {"messages": [HumanMessage(content=query)]}
    final_state = graph.invoke(input_, cfg)
    return _build_result(final_state, thread_id)


def _content_to_text(content: Any) -> str:
    """Normalise an LLM content payload to a plain string.

    Gemini can return content as a list of ``{"type": "text", "text": ...}``
    blocks; this joins them back into a single string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content or "")


def _parse_events(evt: dict) -> list[tuple[str, dict]]:
    """Translate one ``astream_events`` event into orchestrator UI events.

    Returns a list of ``(event_name, payload)`` tuples (usually 0 or 1).
    """
    event = evt.get("event")
    name = evt.get("name", "")
    data = evt.get("data") or {}

    if event == "on_tool_start":
        args = data.get("input") or {}
        if name == "update_todos":
            todos = args.get("todos", [])
            return [("plan", {"todos": todos, "step": len(todos)})]
        return [("tool_start", {"name": name, "args": args})]

    if event == "on_tool_end" and name in _TOOL_NAMES and name != "update_todos":
        output = data.get("output")
        content = getattr(output, "content", None) or output
        return [("tool_end", {"name": name, "output": str(content)[:2000]})]

    if event == "on_chat_model_stream":
        chunk = data.get("chunk")
        token = _content_to_text(getattr(chunk, "content", ""))
        if token:
            return [("token", {"delta": token})]

    return []


async def orchestrator_sse_events(
    graph, query: str, thread_id: str | None = None
):
    """Stream an orchestrator run as (event_name, payload) tuples.

    The graph runs in a dedicated worker thread (its own asyncio loop) so the
    FastAPI event loop is never blocked by the synchronous model/tool calls.
    Events: ``plan``, ``tool_start``, ``tool_end``, ``token``, then ``done``
    (with the final result) or ``error``.

    Args:
        graph: A compiled orchestrator graph.
        query: The user's question.
        thread_id: Conversation id for memory (auto-generated when ``None``).

    Yields:
        ``(event_name, payload)`` tuples.
    """
    cfg = _make_config(thread_id)
    thread_id = cfg["configurable"]["thread_id"]
    input_ = {"messages": [HumanMessage(content=query)]}
    events_queue: "queue.Queue" = queue.Queue()

    def _worker() -> None:
        async def _inner() -> None:
            try:
                async for evt in graph.astream_events(input_, config=cfg, version="v2"):
                    events_queue.put(("raw", evt))
                final_state = graph.get_state(cfg)
                events_queue.put(("state", final_state.values))
            except Exception as exc:  # noqa: BLE001 - forwarded to the client
                logger.exception("Orchestrator run failed")
                events_queue.put(("error", exc))
            finally:
                events_queue.put(None)

        asyncio.new_event_loop().run_until_complete(_inner())

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        item = await asyncio.to_thread(events_queue.get)
        if item is None:
            break
        kind, payload = item
        if kind == "raw":
            for event_name, event_payload in _parse_events(payload):
                yield event_name, event_payload
        elif kind == "error":
            yield "error", {"message": str(payload)}
            return
        else:  # "state"
            yield "done", _build_result(payload, thread_id)
