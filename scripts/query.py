"""Retrieval pipeline entry point.

Run this to load the vector store, retrieve chunks for a query,
and generate a multimodal answer.

Usage:
    python -m scripts.query
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embed import load_embedding_model
from src.retrieval.search import load_vector_store, build_retriever, retrieve_chunks
from src.retrieval.generate import generate_answer


def main() -> None:
    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print("Loading vector store...")
    db = load_vector_store(embedding_model)

    retriever = build_retriever(db)
    print("Ready.")

    query = input("Enter your query: ")
    if not query.strip():
        print("No query provided.")
        return

    print("Running retrieval...")
    chunks = retrieve_chunks(retriever, query)
    print(f"Retrieved {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        print("\n" + "=" * 80)
        print(f"CHUNK {i + 1}")
        print("=" * 80)

        original_data = chunk.metadata.get("original_content")
        if original_data:
            import json
            if isinstance(original_data, str):
                original_data = json.loads(original_data)
            print(original_data.get("raw_text", "")[:1000])

    print("\n" + "-" * 5)
    print("Generating answer...")
    final_answer = generate_answer(chunks, query, verbose=True)
    print("-" * 5)
    print(final_answer)


if __name__ == "__main__":
    main()
