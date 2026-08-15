"""Orchestrator agent.

A LangGraph agent that answers questions by combining the project's internal
knowledge base (Chroma) with live web research (Tavily search + page
extraction).  It follows the "deep agent" pattern:

* a detailed system prompt describing exactly how and when to use each tool,
* a planning tool (``update_todos``) that makes the LLM write down its plan
  before acting,
* a tool-calling loop with a persistence checkpointer so conversation memory
  survives across turns (keyed by ``thread_id``),
* LangSmith tracing for every run.

The public entry points are :func:`build_orchestrator` and the streaming
helper :func:`orchestrator_sse_events`.
"""

from src.orchestrator.agent import (
    build_orchestrator,
    create_orchestrator_llm,
    orchestrator_sse_events,
    run_orchestrator,
)

__all__ = [
    "build_orchestrator",
    "create_orchestrator_llm",
    "orchestrator_sse_events",
    "run_orchestrator",
]
