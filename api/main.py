"""Multimodal RAG HTTP API: the FastAPI application.

This module only assembles the app: it creates it, adds CORS, sets up the
lifespan that holds one ``RAGService``, and includes the routers in
``api/routers/``, where the endpoints live.

Run:
    uvicorn api.main:app --reload --port 8000

Interactive docs live at ``http://localhost:8000/docs``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import images, query, retrieval, system
from api.service import RAGService
from config.settings import CORS_ALLOWED_ORIGINS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Hold one RAG service (loaded lazily on first request)."""
    app.state.service = RAGService()
    yield


app = FastAPI(
    title="Multimodal RAG API",
    version="0.2.0",
    description=(
        "Retrieve chunks and generate multimodal answers from a PDF-based "
        "vector store.\n\n"
        "Every chunk carries its OCR text, extracted tables (HTML) and figures "
        "(inline base64 **and** fetchable image URLs). Try the example questions "
        "like *\"Tell me about the pin configuration of the LM317.\"*"
    ),
    lifespan=lifespan,
)

# Browser access: allow the configured origins (e.g. the local Vite dev server,
# and later the production frontend URL).  Non-browser callers are unaffected.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(retrieval.router)
app.include_router(query.router)
app.include_router(images.router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
