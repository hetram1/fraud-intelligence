from __future__ import annotations

from typing import Any


class EvidenceSynthesisAgent:
    """
    Combines ML, graph, and RAG outputs into a structured,
    evidence-grounded investigation report.

    This stage is deterministic by design. An LLM can be
    added later as a report-generation layer without
    changing the underlying evidence contract.
    """

    def synthesize(
        self,
        claim_id: str,
        question: str,
        fraud_score: dict[str, Any],
        graph_evidence: dict[str, Any],
        rag_evidence: dict[str, Any],
    ) -> dict[str, Any]:

        score = float(
            fraud_score["fraud_risk_score"]
        )

        threshold = float(
            fraud_score["threshold"]
        )

        flagged = bool(
            fraud_score["flagged_at_threshold"]
        )

        relationships = graph_evidence[
            "relationships"
        ]

        attributes = graph_evidence[
            "attributes"
        ]

        shared_claims = relationships.get(
            "shared_location_claims",
            [],
        )

        evidence_sources = [
            item["source_file"]
            for item in rag_evidence.get(
                "evidence",
                [],
            )
        ]

        report = self._build_report(
            claim_id=claim_id,
            question=question,
            score=score,
            threshold=threshold,
            flagged=flagged,
            graph_evidence=graph_evidence,
            rag_evidence=rag_evidence,
        )

        return {
            "claim_id": claim_id,
            "question": question,
            "risk": {
                "fraud_risk_score": score,
                "threshold": threshold,
                "flagged_at_threshold": flagged,
            },
            "graph_summary": {
                "shared_location_claim_count": len(
                    shared_claims
                ),
                "incident_type": attributes.get(
                    "incident_type"
                ),
                "collision_type": attributes.get(
                    "collision_type"
                ),
                "severity": attributes.get(
                    "severity"
                ),
                "incident_state": attributes.get(
                    "incident_state"
                ),
            },
            "rag_summary": {
                "retrieved_chunks": int(
                    rag_evidence.get(
                        "result_count",
                        0,
                    )
                ),
                "sources": sorted(
                    set(evidence_sources)
                ),
            },
            "report": report,
        }

    @staticmethod
    def _build_report(
        claim_id: str,
        question: str,
        score: float,
        threshold: float,
        flagged: bool,
        graph_evidence: dict[str, Any],
        rag_evidence: dict[str, Any],
    ) -> str:

        claim = graph_evidence["claim"] \
            if "claim" in graph_evidence \
            else {}

        attributes = graph_evidence[
            "attributes"
        ]

        relationships = graph_evidence[
            "relationships"
        ]

        location = graph_evidence[
            "location"
        ]

        shared_claims = relationships.get(
            "shared_location_claims",
            [],
        )

        sources = [
            item["source_file"]
            for item in rag_evidence.get(
                "evidence",
                [],
            )
        ]

        sources_text = (
            ", ".join(sorted(set(sources)))
            if sources
            else "No documents retrieved."
        )

        risk_status = (
            "above"
            if flagged
            else "below"
        )

        lines = [
            f"# Investigation Report — {claim_id}",
            "",
            f"Investigation question: {question}",
            "",
            "## Fraud Model Signal",
            (
                f"The model produced a risk score of "
                f"{score:.4f}, which is {risk_status} "
                f"the configured threshold of {threshold:.2f}."
            ),
            (
                "This score is a model-derived risk signal "
                "and is not, by itself, proof of fraud."
            ),
            "",
            "## Claim Evidence",
            (
                f"Incident date: "
                f"{claim.get('incident_date')}"
            ),
            (
                f"Claim type: "
                f"{claim.get('claim_type')}"
            ),
            (
                f"Claimed amount: "
                f"{claim.get('claimed_amount')}"
            ),
            (
                f"Total claim amount: "
                f"{claim.get('total_claim_amount')}"
            ),
            (
                f"Incident location: "
                f"{location.get('city')}, "
                f"{location.get('state')}"
            ),
            (
                f"Police report available: "
                f"{claim.get('police_report_available')}"
            ),
            "",
            "## Graph Evidence",
            (
                f"Incident type: "
                f"{attributes.get('incident_type')}"
            ),
            (
                f"Collision type: "
                f"{attributes.get('collision_type')}"
            ),
            (
                f"Severity: "
                f"{attributes.get('severity')}"
            ),
            (
                f"Claims sharing the incident location: "
                f"{len(shared_claims)}"
            ),
        ]

        if shared_claims:
            lines.append(
                "Related claim IDs: "
                + ", ".join(shared_claims)
            )
        else:
            lines.append(
                "No shared-location claim relationships "
                "were returned for this claim."
            )

        lines.extend(
            [
                "",
                "## Retrieved Guidance",
                (
                    f"Retrieved {rag_evidence.get('result_count', 0)} "
                    f"knowledge chunks."
                ),
                f"Sources: {sources_text}",
                "",
                "## Investigation Interpretation",
                (
                    "The available evidence should be reviewed "
                    "together. Model output, graph relationships, "
                    "and procedural guidance are separate evidence "
                    "types and should not be treated as interchangeable."
                ),
                (
                    "A detected graph relationship or elevated "
                    "model score is an investigation signal rather "
                    "than a definitive fraud determination."
                ),
            ]
        )

        return "\n".join(lines)


def main() -> None:
    print("=" * 70)
    print("EVIDENCE SYNTHESIS AGENT")
    print("=" * 70)

    from src.agents.fraud_scoring_agent import (
        FraudScoringAgent,
    )
    from src.agents.graph_investigation_agent import (
        GraphInvestigationAgent,
    )
    from src.agents.rag_evidence_agent import (
        RAGEvidenceAgent,
    )

    claim_id = "CLM_POL100000"

    question = (
        "Investigate this potentially suspicious claim."
    )

    fraud = FraudScoringAgent().score(
        claim_id
    )

    graph = GraphInvestigationAgent().investigate(
        claim_id
    )

    rag = RAGEvidenceAgent(
        top_k=3
    ).retrieve_evidence(
        claim_id=claim_id,
        question=question,
    )

    result = EvidenceSynthesisAgent().synthesize(
        claim_id=claim_id,
        question=question,
        fraud_score=fraud,
        graph_evidence=graph,
        rag_evidence=rag,
    )

    print("\nREPORT")
    print("-" * 70)
    print(result["report"])

    print("\nEVIDENCE SYNTHESIS AGENT: PASS")


if __name__ == "__main__":
    main()
