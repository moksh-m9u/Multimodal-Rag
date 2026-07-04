"""Orchestrator: ingestion pipeline and vector store creation."""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_METADATA
from src.embed import load_embedding_model
from src.ingestion.extract import partition_document
from src.ingestion.chunk import create_chunks_by_title
from src.ingestion.enrich import summarise_chunks
from src.ingestion.export import export_chunks_to_json


def create_vector_store(
    documents: list[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """Create and persist a ChromaDB vector store from documents.

    Args:
        documents: List of LangChain Documents.
        persist_directory: Directory to persist the vector store.

    Returns:
        The created Chroma vector store.
    """
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = load_embedding_model()

    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata=CHROMA_COLLECTION_METADATA,
    )
    print("--- Finished creating vector store ---")

    print(f"Vector store saved to {persist_directory}")
    return vectorstore


def run_complete_ingestion_pipeline(pdf_path: str) -> Chroma:
    """Run the full RAG ingestion pipeline end-to-end.

    Steps: partition -> chunk -> AI summarise -> export -> vector store.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        The Chroma vector store ready for retrieval.
    """
    print("Starting RAG Ingestion Pipeline")
    print("=" * 50)

    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)
    summarised = summarise_chunks(chunks)
    export_chunks_to_json(summarised, filename="chunks_huggingface.json")
    db = create_vector_store(summarised, persist_directory="dbv2/chroma_db")

    print("Pipeline completed successfully!")
    return db
