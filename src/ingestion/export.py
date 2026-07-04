"""Export processed chunks to JSON."""

import json

from langchain_core.documents import Document


def export_chunks_to_json(
    chunks: list[Document],
    filename: str = "chunks_export.json",
) -> list[dict]:
    """Export processed LangChain Documents to a clean JSON file.

    Each entry contains chunk_id, enhanced_content, and metadata with
    the original raw_text, tables_html, and images_base64.

    Args:
        chunks: List of LangChain Documents.
        filename: Output JSON file path.

    Returns:
        The exported data as a list of dicts.
    """
    export_data: list[dict] = []

    for i, doc in enumerate(chunks):
        raw_meta = doc.metadata.get("original_content", "{}")
        if isinstance(raw_meta, str):
            original = json.loads(raw_meta)
        else:
            original = raw_meta

        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {"original_content": original},
        }
        export_data.append(chunk_data)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(export_data)} chunks to {filename}")
    return export_data
