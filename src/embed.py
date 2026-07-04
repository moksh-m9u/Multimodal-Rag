"""Shared embedding model setup."""

from langchain_huggingface import HuggingFaceEndpointEmbeddings

from config.settings import EMBEDDING_MODEL


def load_embedding_model() -> HuggingFaceEndpointEmbeddings:
    """Return the HuggingFace embedding model."""
    return HuggingFaceEndpointEmbeddings(model=EMBEDDING_MODEL)
