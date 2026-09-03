"""Tests for src/tracing.py trace metadata summarisation."""

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document

from src.tracing import summarize_chunks


def _doc(**metadata):
    return Document(page_content="summary text here", metadata=metadata)


def test_summarize_flat_chunk():
    chunk = _doc(raw_text="raw", tables_html='["<t/>"]', image_paths=[])
    out = summarize_chunks([chunk])[0]
    assert out["enhanced_length"] == len("summary text here")
    assert out["has_table"] is True
    assert out["has_image"] is False
    assert out["table_count"] == 1
    assert out["image_count"] == 0


def test_summarize_empty_chunk():
    chunk = Document(page_content="", metadata={})
    out = summarize_chunks([chunk])[0]
    assert out["enhanced_length"] == 0
    assert out["raw_text_length"] == 0
    assert out["has_table"] is False
    assert out["has_image"] is False


def test_max_preview_caps_chunks():
    chunks = [Document(page_content=str(i), metadata={}) for i in range(10)]
    assert len(summarize_chunks(chunks, max_preview=3)) == 3
