"""State definition for the orchestrator agent.

The agent keeps the full message history (via ``add_messages``), the current
plan of action (``todos``) and a list of ``sources`` it consulted, so the
frontend can render citations and the plan alongside the final answer.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class OrchestratorState(TypedDict):
    """State passed between orchestrator graph nodes."""

    #: Conversation history; ``add_messages`` appends and deduplicates by ID.
    messages: Annotated[list[BaseMessage], add_messages]

    #: The plan of steps the agent wrote down with ``update_todos``.
    todos: list[str]

    #: Citation metadata collected from tool results (list of dicts).
    sources: list[dict]
