"""Vector store loading and document retrieval."""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document

from config.settings import (
    CHROMA_PERSIST_DIR,
    RETRIEVAL_K,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_SEARCH_TYPE,
)


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


def retrieve_chunks(retriever, query: str) -> list[Document]:
    """Run retrieval and return matching chunks."""
    return retriever.invoke(query)
