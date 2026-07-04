"""Ingestion pipeline entry point.

Run this to partition a PDF, chunk, enrich with AI, export to JSON,
and create a Chroma vector store.

Usage:
    python -m scripts.ingest                          # default path
    python -m scripts.ingest path/to/document.pdf     # custom path
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.pipeline import create_vector_store
from src.ingestion.export import export_chunks_to_json
from src.ingestion.extract import partition_document
from src.ingestion.chunk import create_chunks_by_title
from src.ingestion.enrich import summarise_chunks


DEFAULT_PDF_PATH = "./data/datasheet.pdf"


def run_with_retrieval_test(pdf_path: str) -> None:
    """Run pipeline, then do a quick retrieval test."""
    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)
    summarised = summarise_chunks(chunks)

    export_chunks_to_json(summarised, filename="chunks_huggingface.json")
    db = create_vector_store(summarised)

    query = "explain the internal architecture"
    retriever = db.as_retriever(search_kwargs={"k": 5})
    results = retriever.invoke(query)

    export_chunks_to_json(results, "rag_results.json")
    print(f"\nDone. Retrieved {len(results)} chunks for query: {query}")


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_PATH

    print("=" * 60)
    print("Multimodal RAG Ingestion Pipeline")
    print("=" * 60)

    run_with_retrieval_test(pdf)
