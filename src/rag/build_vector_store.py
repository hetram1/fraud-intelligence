from __future__ import annotations

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]

CHUNK_PATH = (
    ROOT / "data" / "rag" / "chunks" / "chunks.jsonl"
)

VECTOR_DIR = ROOT / "outputs" / "rag" / "chroma"

COLLECTION_NAME = "fraud_knowledge"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks() -> list[dict]:
    if not CHUNK_PATH.exists():
        raise FileNotFoundError(
            f"Missing chunk file: {CHUNK_PATH}"
        )

    chunks = []

    with CHUNK_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                chunks.append(json.loads(line))

    if not chunks:
        raise RuntimeError("No chunks found.")

    return chunks


def main() -> None:
    print("=" * 70)
    print("RAG VECTOR STORE BUILD")
    print("=" * 70)

    chunks = load_chunks()

    print(f"Chunks loaded: {len(chunks)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "document_id": int(chunk["document_id"]),
            "source_file": chunk["source_file"],
            "chunk_index": int(chunk["chunk_index"]),
        }
        for chunk in chunks
    ]

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("[OK] Embedding model loaded.")

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    print(
        f"[OK] Generated embeddings: "
        f"{embeddings.shape}"
    )

    VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nCreating ChromaDB collection...")

    client = chromadb.PersistentClient(
        path=str(VECTOR_DIR)
    )

    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
        print("[OK] Removed existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
        },
    )

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    stored_count = collection.count()

    print(
        f"[OK] Stored documents/chunks: "
        f"{stored_count}"
    )

    if stored_count != len(chunks):
        raise RuntimeError(
            f"Vector store count mismatch: "
            f"expected {len(chunks)}, "
            f"got {stored_count}"
        )

    metadata = {
        "collection": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_count": len(chunks),
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "distance_metric": "cosine",
        "path": str(VECTOR_DIR),
    }

    metadata_path = (
        ROOT
        / "outputs"
        / "rag"
        / "vector_store_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"\nSaved metadata: {metadata_path}"
    )

    print("\n" + "=" * 70)
    print("RAG VECTOR STORE BUILD: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
