from __future__ import annotations

from typing import Any

from src.rag.retriever import RAGRetriever


class RAGEvidenceAgent:
    """
    RAG specialist responsible only for retrieving
    relevant knowledge-base evidence.
    """

    def __init__(self, top_k: int = 3) -> None:
        self.retriever = RAGRetriever(top_k=top_k)

    def retrieve_evidence(
        self,
        claim_id: str,
        question: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:

        if not claim_id.strip():
            raise ValueError(
                "claim_id must not be empty."
            )

        if not question.strip():
            raise ValueError(
                "question must not be empty."
            )

        results = self.retriever.retrieve(
            question,
            top_k=top_k,
        )

        evidence = []

        for rank, result in enumerate(
            results,
            start=1,
        ):
            evidence.append(
                {
                    "rank": rank,
                    "source_file": result[
                        "source_file"
                    ],
                    "chunk_index": result[
                        "chunk_index"
                    ],
                    "distance": result[
                        "distance"
                    ],
                    "text": result["text"],
                }
            )

        return {
            "claim_id": claim_id,
            "question": question,
            "result_count": len(evidence),
            "evidence": evidence,
        }


def main() -> None:
    print("=" * 70)
    print("RAG EVIDENCE AGENT")
    print("=" * 70)

    agent = RAGEvidenceAgent(top_k=3)

    result = agent.retrieve_evidence(
        claim_id="CLM_POL100000",
        question=(
            "How should this potentially suspicious "
            "insurance claim be reviewed?"
        ),
    )

    print(f"\nClaim: {result['claim_id']}")
    print(f"Question: {result['question']}")
    print(f"Retrieved chunks: {result['result_count']}")

    print("\nRETRIEVED EVIDENCE")
    print("-" * 70)

    for item in result["evidence"]:
        print(
            f"\n[{item['rank']}] "
            f"{item['source_file']} "
            f"(chunk {item['chunk_index']})"
        )

        print(
            f"Distance: {item['distance']:.6f}"
        )

        print(
            item["text"][:450]
        )

    print("\nRAG EVIDENCE AGENT: PASS")


if __name__ == "__main__":
    main()
