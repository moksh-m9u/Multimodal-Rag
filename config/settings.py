"""Central configuration and environment variable loading.

All tunable knobs for the pipeline live here so they can be changed from a
single place.  Secrets are read from the ``.env`` file (loaded once at import)
and may be overridden by real environment variables.

Module docstrings keep the file self-explanatory; see each section below for
the pipeline stage it controls.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

# Load secrets from a .env file if present (safe to call multiple times).
load_dotenv()

# ---------------------------------------------------------------------------
# API credentials (loaded from environment / .env)
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
HUGGINGFACEHUB_API_TOKEN: str = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
# Embedding model used to vectorise chunks during ingestion and retrieval.
EMBEDDING_MODEL: str = "ibm-granite/granite-embedding-97m-multilingual-r2"
# Multimodal vision LLM used for AI-enhanced chunk summaries during ingestion.
ENHANCEMENT_MODEL: str = "zai-org/GLM-4.5V"
# Router endpoint that serves the enhancement model via the OpenAI protocol.
ENHANCEMENT_BASE_URL: str = "https://router.huggingface.co/v1"
# Text generation model used to answer retrieval queries.
GENERATION_MODEL: str = "gemini-2.5-flash-lite"

# ---------------------------------------------------------------------------
# Storage locations
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = "dbv2/chroma_db"
IMAGES_DIR: str = "dbv2/images"

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
RETRIEVAL_K: int = 10          # Number of chunks to return.
RETRIEVAL_FETCH_K: int = 20    # Candidate pool before MMR re-ranking.
RETRIEVAL_SEARCH_TYPE: str = "mmr"

# ---------------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------------
GENERATION_TEMPERATURE: float = 0.0

# ---------------------------------------------------------------------------
# PDF extraction settings (unstructured partition_pdf)
# ---------------------------------------------------------------------------
PDF_STRATEGY: str = "hi_res"
PDF_EXTRACT_IMAGE_BLOCK_TYPES: list[str] = ["Image"]
PDF_INFER_TABLE_STRUCTURE: bool = True

# ---------------------------------------------------------------------------
# Chunking settings (unstructured chunk_by_title)
# ---------------------------------------------------------------------------
CHUNK_MAX_CHARACTERS: int = 3000
CHUNK_NEW_AFTER_N_CHARS: int = 2400
CHUNK_COMBINE_UNDER_N_CHARS: int = 500
CHUNK_ISOLATE_TABLES: bool = False

# ---------------------------------------------------------------------------
# Enhancement (AI summary) settings
# ---------------------------------------------------------------------------
ENHANCEMENT_TEMPERATURE: float = 0.0
ENHANCEMENT_MAX_TOKENS: int = 1024

# ---------------------------------------------------------------------------
# Vector store settings
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_METADATA: dict = {"hnsw:space": "cosine"}
