"""Embedding model factory.

Provides a single function to build the embedding model used by both the
ingestion pipeline (when creating the vector store) and the retrieval path
(when loading it back), guaranteeing both sides use the same configuration.

Embeddings are served by IBM watsonx.ai, so the store must be re-indexed
whenever ``EMBEDDING_MODEL`` changes: vectors from different models are not
comparable and usually differ in dimension.
"""

from __future__ import annotations

from langchain_ibm import WatsonxEmbeddings

from config.settings import (
    EMBEDDING_MAX_INPUT_TOKENS,
    EMBEDDING_MODEL,
    WATSONX_API_KEY,
    WATSONX_PROJECT_ID,
    WATSONX_URL,
)
from src.logger import get_logger

logger = get_logger(__name__)


def load_embedding_model() -> WatsonxEmbeddings:
    """Create the watsonx.ai embedding model configured in settings.

    Inputs longer than ``EMBEDDING_MAX_INPUT_TOKENS`` are truncated by the
    service; without this, watsonx rejects the whole request with a 400.

    Returns:
        A configured :class:`WatsonxEmbeddings` instance ready to embed text
        or load a vector store.

    Raises:
        RuntimeError: If ``WATSONX_API_KEY`` or ``WATSONX_PROJECT_ID`` is not
            set, so the failure names the missing variable instead of
            surfacing as an opaque authentication error.
    """
    missing = [
        name
        for name, value in (
            ("WATSONX_API_KEY", WATSONX_API_KEY),
            ("WATSONX_PROJECT_ID", WATSONX_PROJECT_ID),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing watsonx.ai credentials: {', '.join(missing)}")

    logger.info("Loading embedding model: %s (%s)", EMBEDDING_MODEL, WATSONX_URL)
    return WatsonxEmbeddings(
        model_id=EMBEDDING_MODEL,
        url=WATSONX_URL,
        project_id=WATSONX_PROJECT_ID,
        api_key=WATSONX_API_KEY,
        params={"truncate_input_tokens": EMBEDDING_MAX_INPUT_TOKENS},
    )
