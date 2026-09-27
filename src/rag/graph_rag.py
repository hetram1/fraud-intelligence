from __future__ import annotations

import argparse
from typing import Any

from src.graph.neo4j_client import get_driver
from src.rag.retriever import RAGRetriever


class GraphRAGEvidenceRetriever:
    def __init__(self, rag_top_k: int = 3) -> None:
        self.rag = RAGRetriever(top_k=rag_top_k)

    def retrieve_graph_evidence(
        self,
        claim_id: str,
    ) -> dict[str, Any]:
        driver = get_driver()

        try:
            with driver.session() as session:
                record = session.run(
                    """
                    MATCH (c:Claim {claim_id: $claim_id})

                    OPTIONAL MATCH (p:Policy)-[:HAS_CLAIM]->(c)

                    OPTIONAL MATCH (c)-[:OCCURRED_AT]->(location)

                    OPTIONAL MATCH
                        (c)-[:SHARES_LOCATION_WITH]-(neighbor:Claim)

                    OPTIONAL MATCH
                        (c)-[:HAS_INCIDENT_TYPE]->(incident_type)

                    OPTIONAL MATCH
                        (c)-[:HAS_COLLISION_TYPE]->(collision_type)

                    OPTIONAL MATCH
                        (c)-[:HAS_SEVERITY]->(severity)

                    OPTIONAL MATCH
                        (c)-[:HAS_INCIDENT_STATE]->(incident_state)

                    RETURN
                        c.claim_id AS claim_id,
                        c.incident_date AS incident_date,
                        c.claim_type AS claim_type,
                        c.claimed_amount AS claimed_amount,
                        c.total_claim_amount AS total_claim_amount,
                        c.incident_city AS incident_city,
                        c.incident_state AS incident_state,
                        c.incident_hour_of_day AS incident_hour_of_day,
                        c.number_of_vehicles_involved
                            AS number_of_vehicles_involved,
                        c.bodily_injuries AS bodily_injuries,
                        c.witnesses AS witnesses,
                        c.police_report_available
                            AS police_report_available,

                        p.policy_id AS policy_id,
                        p.premium AS policy_premium,
                        p.deductible AS policy_deductible,

                        location.location_key AS location_key,

                        collect(DISTINCT neighbor.claim_id)[0..10]
                            AS related_claim_ids,

                        incident_type.name
                            AS incident_type_node,

                        collision_type.name
                            AS collision_type_node,

                        severity.name
                            AS severity_node,

                        incident_state.name
                            AS incident_state_node
                    """,
                    claim_id=claim_id,
                ).single()

                if record is None:
                    raise ValueError(
                        f"Claim not found in Neo4j: {claim_id}"
                    )

                return {
                    "claim": {
                        "claim_id": record["claim_id"],
                        "incident_date": record["incident_date"],
                        "claim_type": record["claim_type"],
                        "claimed_amount": record["claimed_amount"],
                        "total_claim_amount": record[
                            "total_claim_amount"
                        ],
                        "incident_city": record[
                            "incident_city"
                        ],
                        "incident_state": record[
                            "incident_state"
                        ],
                        "incident_hour_of_day": record[
                            "incident_hour_of_day"
                        ],
                        "number_of_vehicles_involved": record[
                            "number_of_vehicles_involved"
                        ],
                        "bodily_injuries": record[
                            "bodily_injuries"
                        ],
                        "witnesses": record["witnesses"],
                        "police_report_available": record[
                            "police_report_available"
                        ],
                    },
                    "policy": {
                        "policy_id": record["policy_id"],
                        "premium": record[
                            "policy_premium"
                        ],
                        "deductible": record[
                            "policy_deductible"
                        ],
                    },
                    "location": {
                        "location_key": record[
                            "location_key"
                        ],
                        "city": record["incident_city"],
                        "state": record["incident_state"],
                    },
                    "relationships": {
                        "shared_location_claims": (
                            record["related_claim_ids"]
                        ),
                        "incident_type": record[
                            "incident_type_node"
                        ],
                        "collision_type": record[
                            "collision_type_node"
                        ],
                        "severity": record["severity_node"],
                        "incident_state": record[
                            "incident_state_node"
                        ],
                    },
                }

        finally:
            driver.close()

    def retrieve(
        self,
        claim_id: str,
        question: str,
        rag_top_k: int | None = None,
    ) -> dict[str, Any]:
        graph_evidence = self.retrieve_graph_evidence(
            claim_id
        )

        rag_evidence = self.rag.retrieve(
            question,
            top_k=rag_top_k,
        )

        return {
            "claim_id": claim_id,
            "question": question,
            "graph_evidence": graph_evidence,
            "rag_evidence": rag_evidence,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve combined Graph-RAG evidence."
    )

    parser.add_argument(
        "--claim-id",
        required=True,
        help="Claim ID to investigate.",
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Investigation question.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of RAG chunks to retrieve.",
    )

    args = parser.parse_args()

    retriever = GraphRAGEvidenceRetriever(
        rag_top_k=args.top_k
    )

    result = retriever.retrieve(
        claim_id=args.claim_id,
        question=args.question,
    )

    print("=" * 70)
    print("GRAPH-RAG EVIDENCE RETRIEVAL")
    print("=" * 70)

    print("\nCLAIM")
    print("-" * 70)

    for key, value in result[
        "graph_evidence"
    ]["claim"].items():
        print(f"{key}: {value}")

    print("\nPOLICY")
    print("-" * 70)

    for key, value in result[
        "graph_evidence"
    ]["policy"].items():
        print(f"{key}: {value}")

    print("\nGRAPH RELATIONSHIPS")
    print("-" * 70)

    for key, value in result[
        "graph_evidence"
    ]["relationships"].items():
        print(f"{key}: {value}")

    print("\nRAG EVIDENCE")
    print("-" * 70)

    for index, item in enumerate(
        result["rag_evidence"],
        start=1,
    ):
        print(
            f"\n[{index}] "
            f"{item['source_file']} "
            f"(chunk {item['chunk_index']})"
        )

        print(
            f"Distance: {item['distance']:.6f}"
        )

        print(item["text"][:350])

    print("\nGRAPH-RAG RETRIEVAL: PASS")


if __name__ == "__main__":
    main()
