"""Index and health endpoints.

``/`` lists what the API exposes and ``/health`` reports whether the vector
store loads. Both stay unversioned so uptime probes have a fixed URL.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["meta"])


@router.get("/", summary="List endpoints")
def root(request: Request) -> dict:
    """Human-readable index of what this API exposes."""
    return {
        "name": "Multimodal RAG API",
        "version": request.app.version,
        "docs": "/docs",
        "endpoints": {
            "POST /retrieve": "Retrieve chunks only (text, tables, images)",
            "POST /query": "Retrieve chunks and generate a full answer",
            "GET /query/stream": "SSE stream of the answer with retrieval summary",
            "GET /images/{response_id}/{index}": "Fetch a single figure as JPEG",
            "GET /health": "Service health / vector store status",
        },
    }


@router.get("/health", summary="Service health", response_description="Whether the vector store is loadable here")
async def health(request: Request) -> dict:
    """Report whether the vector store is loadable in this runtime.

    Useful as a startup probe: the first call may take a few seconds while the
    embedding model + Chroma store load into memory.
    """
    try:
        await request.app.state.service.ensure_ready()
        return {"status": "ok", "vector_store": "loaded"}
    except Exception as exc:
        return {"status": "error", "vector_store": "unavailable", "detail": str(exc)}
