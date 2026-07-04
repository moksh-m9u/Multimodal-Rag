"""Title-based chunking of unstructured elements."""

from unstructured.chunking.title import chunk_by_title

from config.settings import (
    CHUNK_MAX_CHARACTERS,
    CHUNK_NEW_AFTER_N_CHARS,
    CHUNK_COMBINE_UNDER_N_CHARS,
    CHUNK_ISOLATE_TABLES,
)


def create_chunks_by_title(elements: list) -> list:
    """Create intelligent chunks using a title-based strategy.

    Args:
        elements: Parsed PDF elements from partition_document().

    Returns:
        List of unstructured chunks.
    """
    print("Creating smart chunks...")

    chunks = chunk_by_title(
        elements,
        max_characters=CHUNK_MAX_CHARACTERS,
        new_after_n_chars=CHUNK_NEW_AFTER_N_CHARS,
        combine_text_under_n_chars=CHUNK_COMBINE_UNDER_N_CHARS,
        isolate_table=CHUNK_ISOLATE_TABLES,
    )

    print(f"Created {len(chunks)} chunks")
    return chunks
