import json
import base64
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings


def ingest_json_directory(
    json_directory: str = '../json',
    persist_directory: str = "../dbv2/chroma_db",
    images_directory: str = "../dbv2/images",
):
    """
    Reads all JSON files from a directory and ingests them into ChromaDB.
    Images are saved as separate files to avoid bloating ChromaDB metadata.
    """

    print("=" * 50)
    print("INGESTION PIPELINE START")
    print("=" * 50)

    print("\n[1/4] Loading embedding model...")
    embedding_model = HuggingFaceEndpointEmbeddings(
        model="ibm-granite/granite-embedding-97m-multilingual-r2",
    )
    print(f"  Model: {embedding_model.model}")

    print("\n[2/4] Preparing documents and images...")
    images_dir = Path(images_directory)
    images_dir.mkdir(parents=True, exist_ok=True)

    documents = []

    json_files = sorted(Path(json_directory).glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {json_directory}")

    print(f"  Found {len(json_files)} JSON files")

    total_images = 0
    total_tables = 0
    total_raw_chars = 0

    for json_file in json_files:
        print(f"\n  Processing: {json_file.name}")

        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        print(f"    Chunks in file: {len(chunks)}")

        for chunk in chunks:

            original = chunk["metadata"]["original_content"]
            source_stem = json_file.stem
            chunk_id = chunk["chunk_id"]

            images_b64 = original.get("images_base64", [])
            tables = original.get("tables_html", [])
            raw_text = original.get("raw_text", "")

            image_paths = []
            for idx, b64 in enumerate(images_b64):
                img_filename = f"{source_stem}_c{chunk_id}_img{idx}.jpg"
                img_path = images_dir / img_filename
                img_path.write_bytes(base64.b64decode(b64))
                image_paths.append(str(img_path))

            total_images += len(images_b64)
            total_tables += len(tables)
            total_raw_chars += len(raw_text)

            metadata = {
                "source": json_file.stem,
                "chunk_id": chunk_id,
                "raw_text": raw_text,
                "tables_html": json.dumps(tables, ensure_ascii=False),
                "image_paths": json.dumps(image_paths, ensure_ascii=False),
                "has_table": len(tables) > 0,
                "has_image": len(images_b64) > 0,
            }

            documents.append(
                Document(
                    page_content=chunk["enhanced_content"],
                    metadata=metadata,
                )
            )

    print(f"\n  Document summary:")
    print(f"    Total chunks: {len(documents)}")
    print(f"    Total images saved: {total_images}")
    print(f"    Total tables: {total_tables}")
    print(f"    Total raw text chars: {total_raw_chars:,}")

    print(f"\n[3/4] Generating embeddings ({len(documents)} chunks)...")
    print("  This may take a while depending on API quota...")

    t0 = time.time()
    try:
        db = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )
        elapsed = time.time() - t0
        print(f"  Embeddings completed in {elapsed:.1f}s")
        print(f"  Collection size: {db._collection.count()} documents")

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}")

        if "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower():
            print("\n  -> QUOTA/RATE LIMIT EXCEEDED")
            print("  -> Wait a few minutes and re-run this cell.")
            print("  -> Or switch to a local embedding model (e.g. sentence-transformers).")
        elif "401" in str(e) or "403" in str(e) or "auth" in str(e).lower():
            print("\n  -> AUTH ERROR - check your HF_TOKEN")
        elif "readonly" in str(e).lower():
            print("\n  -> DATABASE LOCKED - delete the persist_directory and re-run.")
        else:
            print(f"\n  -> Unexpected error, check the traceback above.")

        raise

    print(f"\n[4/4] Verifying...")
    count = db._collection.count()
    print(f"  Documents in ChromaDB: {count}")
    if count != len(documents):
        print(f"  WARNING: Expected {len(documents)} but got {count}")

    print(f"\n  Images saved to: {images_dir.resolve()}")
    print(f"  DB saved to: {Path(persist_directory).resolve()}")

    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)

    return db

ingest_json_directory()
