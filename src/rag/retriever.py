from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]

VECTOR_DIR = ROOT / "outputs" / "rag" / "chroma"

COLLECTION_NAME = "fraud_knowledge"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RAGRetriever:
    def __init__(
        self,
        top_k: int = 3,
    ) -> None:
        self.top_k = top_k

        if not VECTOR_DIR.exists():
            raise FileNotFoundError(
                f"Vector store not found: {VECTOR_DIR}"
            )

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DIR)
        )

        self.collection = self.client.get_collection(
            COLLECTION_NAME
        )

        if self.collection.count() == 0:
            raise RuntimeError(
                "RAG collection is empty."
            )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:

        query = query.strip()

        if not query:
            raise ValueError(
                "Query must not be empty."
            )

        k = top_k or self.top_k

        if k < 1:
            raise ValueError(
                "top_k must be >= 1."
            )

        query_embedding = (
            self.embedding_model.encode(
                [query],
                normalize_embeddings=True,
            )
        )

        result = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        retrieved = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            retrieved.append(
                {
                    "text": document,
                    "source_file": metadata[
                        "source_file"
                    ],
                    "chunk_index": int(
                        metadata["chunk_index"]
                    ),
                    "distance": float(distance),
                }
            )

        return retrieved


def main() -> None:
    print("=" * 70)
    print("RAG RETRIEVER")
    print("=" * 70)

    retriever = RAGRetriever(top_k=3)

    query = (
        "How should an investigator review a "
        "potentially suspicious insurance claim?"
    )

    results = retriever.retrieve(query)

    print(f"Query: {query}")
    print(f"Results: {len(results)}")

    print("\nRETRIEVED EVIDENCE")
    print("-" * 70)

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n[{index}] "
            f"{result['source_file']} "
            f"(chunk {result['chunk_index']})"
        )

        print(
            f"Distance: "
            f"{result['distance']:.6f}"
        )

        print(
            result["text"][:500]
        )

    print("\nRAG RETRIEVAL: PASS")


if __name__ == "__main__":
    main()
