from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "rag" / "raw"
CHUNK_DIR = ROOT / "data" / "rag" / "chunks"
OUTPUT_PATH = CHUNK_DIR / "chunks.jsonl"
METADATA_PATH = CHUNK_DIR / "metadata.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str) -> list[str]:
    text = normalize_text(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def main() -> None:
    print("=" * 70)
    print("RAG DOCUMENT CHUNKING")
    print("=" * 70)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"RAG raw directory not found: {RAW_DIR}"
        )

    files = sorted(RAW_DIR.glob("*.md"))

    if not files:
        raise RuntimeError(
            f"No Markdown documents found in {RAW_DIR}"
        )

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    document_stats = []

    for document_id, path in enumerate(files):
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        if not chunks:
            raise RuntimeError(
                f"No chunks generated for {path.name}"
            )

        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "chunk_id": (
                        f"{path.stem}_chunk_{chunk_index:04d}"
                    ),
                    "document_id": document_id,
                    "source_file": path.name,
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )

        document_stats.append(
            {
                "source_file": path.name,
                "characters": len(text),
                "chunks": len(chunks),
            }
        )

        print(
            f"[OK] {path.name:<40}"
            f"{len(chunks):>3} chunks"
        )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in all_chunks:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    metadata = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "document_count": len(files),
        "chunk_count": len(all_chunks),
        "documents": document_stats,
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\nSUMMARY")
    print("-" * 70)
    print(f"Documents: {len(files)}")
    print(f"Chunks:    {len(all_chunks)}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Overlap:    {CHUNK_OVERLAP}")

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Saved: {METADATA_PATH}")

    print("\nRAG CHUNKING: PASS")


if __name__ == "__main__":
    main()
