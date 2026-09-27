from __future__ import annotations

from typing import Any

from src.agents.routed_investigation_workflow import (
    build_full_investigation_workflow,
)
from src.agents.llm_report_agent import LLMReportAgent


def build_llm_investigation(question: str, claim_id: str) -> dict[str, Any]:
    """Run the existing deterministic investigation workflow and synthesize it with an LLM."""

    workflow = build_full_investigation_workflow()

    investigation = workflow.invoke(
        {
            "question": question,
            "claim_id": claim_id,
        }
    )

    evidence = {
        "claim_id": claim_id,
        "fraud_score": investigation.get("fraud_score"),
        "graph_evidence": investigation.get("graph_evidence"),
        "rag_evidence": investigation.get("rag_evidence"),
    }

    report_agent = LLMReportAgent()

    return {
        "question": question,
        "claim_id": claim_id,
        "route": investigation.get("route"),
        "evidence": evidence,
        "report": report_agent.generate(evidence),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run full routed investigation and generate an LLM report."
    )
    parser.add_argument(
        "--claim-id",
        default="CLM_POL100000",
    )
    parser.add_argument(
        "--question",
        default="Investigate this claim for potential fraud risk.",
    )
    args = parser.parse_args()

    result = build_llm_investigation(
        question=args.question,
        claim_id=args.claim_id,
    )

    print("=" * 70)
    print("LLM-POWERED FRAUD INVESTIGATION")
    print("=" * 70)
    print(json.dumps(result, indent=2, default=str))
