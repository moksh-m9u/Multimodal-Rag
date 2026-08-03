"""Vector store loading and document retrieval."""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from langsmith import traceable

from config.settings import (
    CHROMA_PERSIST_DIR,
    RETRIEVAL_K,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_SEARCH_TYPE,
)
from src.tracing import summarize_chunks


def load_vector_store(
    embedding_model: HuggingFaceEndpointEmbeddings,
) -> Chroma:
    """Load the persisted Chroma vector store."""
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_model,
    )


def build_retriever(db: Chroma):
    """Build an MMR retriever from the vector store."""
    return db.as_retriever(
        search_type=RETRIEVAL_SEARCH_TYPE,
        search_kwargs={
            "k": RETRIEVAL_K,
            "fetch_k": RETRIEVAL_FETCH_K,
        },
    )


@traceable(run_type="retriever", name="RetrieveChunks")
def retrieve_chunks(retriever, query: str) -> list[Document]:
    """Run retrieval and return matching chunks."""
    chunks = retriever.invoke(query)
    chunk_summaries = summarize_chunks(chunks)
    total_images = sum(c.get("image_count", 0) for c in chunk_summaries)
    total_tables = sum(c.get("table_count", 0) for c in chunk_summaries)
    print(
        f"  Retrieved {len(chunks)} chunks "
        f"({total_images} images, {total_tables} tables)"
    )
    return chunks
