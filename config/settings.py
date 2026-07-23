"""Configuration and environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
HUGGINGFACEHUB_API_TOKEN: str = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

EMBEDDING_MODEL: str = "ibm-granite/granite-embedding-97m-multilingual-r2"
VISION_MODEL: str = "zai-org/GLM-4.5V"
GENERATION_MODEL: str = "gemini-2.5-flash"

CHROMA_PERSIST_DIR: str = "dbv2/chroma_db"
IMAGES_DIR: str = "dbv2/images"

RETRIEVAL_K: int = 10
RETRIEVAL_FETCH_K: int = 20
RETRIEVAL_SEARCH_TYPE: str = "mmr"

GENERATION_TEMPERATURE: float = 0.0
GENERATION_MAX_TOKENS: int = 1024

# Ingestion pipeline
PDF_STRATEGY: str = "hi_res"
PDF_EXTRACT_IMAGE_BLOCK_TYPES: list[str] = ["Image"]
PDF_INFER_TABLE_STRUCTURE: bool = True

CHUNK_MAX_CHARACTERS: int = 3000
CHUNK_NEW_AFTER_N_CHARS: int = 2400
CHUNK_COMBINE_UNDER_N_CHARS: int = 500
CHUNK_ISOLATE_TABLES: bool = False

ENHANCEMENT_MODEL: str = "zai-org/GLM-4.5V"
ENHANCEMENT_BASE_URL: str = "https://router.huggingface.co/v1"
ENHANCEMENT_TEMPERATURE: float = 0.0
ENHANCEMENT_MAX_TOKENS: int = 1024

CHROMA_COLLECTION_METADATA: dict = {"hnsw:space": "cosine"}
