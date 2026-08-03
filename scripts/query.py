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
from src.retrieval.search import load_vector_store, build_retriever
from src.retrieval.generate import answer_query


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

    final_answer = answer_query(retriever, query)
    print("\n" + "-" * 5)
    print(final_answer)


if __name__ == "__main__":
    main()
