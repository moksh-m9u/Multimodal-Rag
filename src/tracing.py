"""LangSmith tracing helpers for multimodal observability."""

import json
from typing import Any

from langchain_core.documents import Document


def _extract_chunk_summary(chunk: Document) -> dict[str, Any]:
    """Build a summary of a chunk's multimodal content for trace metadata."""
    enhanced = chunk.page_content or ""
    meta = chunk.metadata or {}

    raw = meta.get("original_content")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) or {}
        except json.JSONDecodeError:
            raw = {}
    if not isinstance(raw, dict):
        raw = {}

    tables = raw.get("tables_html", [])
    images = raw.get("images_base64", [])
    raw_text = raw.get("raw_text", "")

    if not raw:
        tables_raw = meta.get("tables_html", "[]")
        tables = json.loads(tables_raw) if isinstance(tables_raw, str) else (tables_raw or [])
        image_paths_raw = meta.get("image_paths", "[]")
        images = json.loads(image_paths_raw) if isinstance(image_paths_raw, str) else (image_paths_raw or [])
        raw_text = meta.get("raw_text", "")

    return {
        "enhanced_length": len(enhanced),
        "enhanced_preview": enhanced[:300],
        "raw_text_length": len(raw_text),
        "table_count": len(tables),
        "image_count": len(images),
        "has_table": len(tables) > 0,
        "has_image": len(images) > 0,
    }


def summarize_chunks(chunks: list[Document], max_preview: int = 5) -> list[dict]:
    """Summarize retrieved chunks for trace logging."""
    out = []
    for i, c in enumerate(chunks):
        if i >= max_preview:
            break
        out.append(_extract_chunk_summary(c))
    return out
