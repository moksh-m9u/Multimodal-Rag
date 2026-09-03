"""Tests for src/documents.py schema normalisation."""

import base64

import pytest

langchain_core = pytest.importorskip("langchain_core")

from langchain_core.documents import Document

from src.documents import extract_original_data


def _doc(metadata):
    return Document(page_content="summary", metadata=metadata)


def test_nested_schema_parses_json_payload():
    payload = {
        "raw_text": "hello",
        "tables_html": ["<table></table>"],
        "images_base64": ["abc"],
    }
    doc = _doc({"original_content": payload})
    out = extract_original_data(doc)
    assert out["raw_text"] == "hello"
    assert out["tables_html"] == ["<table></table>"]
    assert out["images_base64"] == ["abc"]


def test_nested_schema_accepts_json_string_and_list_tables():
    doc = _doc({"original_content": '{"raw_text": "hi", "tables_html": ["<t/>"]}'})
    out = extract_original_data(doc)
    assert out["raw_text"] == "hi"
    assert out["tables_html"] == ["<t/>"]
    assert out["images_base64"] == []


def test_flattened_schema_falls_back_to_metadata():
    doc = _doc({"raw_text": "flat", "tables_html": '["<t/>"]', "image_paths": []})
    out = extract_original_data(doc)
    assert out["raw_text"] == "flat"
    assert out["tables_html"] == ["<t/>"]
    assert out["images_base64"] == []


def test_flat_metadata_without_optional_keys():
    out = extract_original_data(_doc({}))
    assert out["raw_text"] == ""
    assert out["tables_html"] == []
    assert out["images_base64"] == []


def test_image_paths_read_from_disk(tmp_path):
    img = base64.b64encode(b"jpeg-bytes").decode()
    p = tmp_path / "fig.jpg"
    p.write_bytes(b"jpeg-bytes")

    # metadata image_paths resolve relative to the repo root; monkey-substitute
    # _PROJECT_ROOT so the tmp file is found.
    import src.documents as documents

    original = documents._PROJECT_ROOT
    documents._PROJECT_ROOT = tmp_path
    try:
        doc = _doc({"raw_text": "", "image_paths": ["fig.jpg"]})
        out = extract_original_data(doc)
    finally:
        documents._PROJECT_ROOT = original

    assert out["images_base64"] == [img]


def test_missing_image_path_is_tolerated():
    doc = _doc({"raw_text": "", "image_paths": ["does_not_exist.jpg"]})
    out = extract_original_data(doc)
    assert out["images_base64"] == [""]
