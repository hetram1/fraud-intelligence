from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.fraud_scoring_agent import (
    FraudScoringAgent,
)
from src.agents.graph_investigation_agent import (
    GraphInvestigationAgent,
)
from src.agents.rag_evidence_agent import (
    RAGEvidenceAgent,
)


class InvestigationState(TypedDict, total=False):
    claim_id: str
    question: str
    fraud_score: dict[str, Any]
    graph_evidence: dict[str, Any]
    rag_evidence: dict[str, Any]


fraud_agent = FraudScoringAgent()
graph_agent = GraphInvestigationAgent()
rag_agent = RAGEvidenceAgent(top_k=3)


def fraud_node(
    state: InvestigationState,
) -> InvestigationState:

    result = fraud_agent.score(
        state["claim_id"]
    )

    return {
        **state,
        "fraud_score": result,
    }


def graph_node(
    state: InvestigationState,
) -> InvestigationState:

    result = graph_agent.investigate(
        state["claim_id"]
    )

    return {
        **state,
        "graph_evidence": result,
    }


def rag_node(
    state: InvestigationState,
) -> InvestigationState:

    result = rag_agent.retrieve_evidence(
        claim_id=state["claim_id"],
        question=state["question"],
    )

    return {
        **state,
        "rag_evidence": result,
    }


def build_workflow():
    workflow = StateGraph(
        InvestigationState
    )

    workflow.add_node(
        "fraud_scoring",
        fraud_node,
    )

    workflow.add_node(
        "graph_investigation",
        graph_node,
    )

    workflow.add_node(
        "rag_evidence",
        rag_node,
    )

    workflow.add_edge(
        START,
        "fraud_scoring",
    )

    workflow.add_edge(
        "fraud_scoring",
        "graph_investigation",
    )

    workflow.add_edge(
        "graph_investigation",
        "rag_evidence",
    )

    workflow.add_edge(
        "rag_evidence",
        END,
    )

    return workflow.compile()


def main() -> None:
    print("=" * 70)
    print("MULTI-AGENT INVESTIGATION WORKFLOW")
    print("=" * 70)

    workflow = build_workflow()

    claim_id = "CLM_POL100000"

    question = (
        "How should this potentially suspicious "
        "insurance claim be reviewed?"
    )

    initial_state: InvestigationState = {
        "claim_id": claim_id,
        "question": question,
    }

    result = workflow.invoke(
        initial_state
    )

    print("\nWORKFLOW RESULT")
    print("-" * 70)

    print(
        f"Claim ID: "
        f"{result['claim_id']}"
    )

    print("\nFraud score:")
    for key, value in result[
        "fraud_score"
    ].items():
        print(f"  {key}: {value}")

    print("\nGraph evidence:")
    print(
        f"  Shared-location claims: "
        f"{result['graph_evidence']['relationships']['shared_location_claim_count']}"
    )

    print(
        f"  Incident type: "
        f"{result['graph_evidence']['attributes']['incident_type']}"
    )

    print(
        f"  Severity: "
        f"{result['graph_evidence']['attributes']['severity']}"
    )

    print("\nRAG evidence:")
    print(
        f"  Retrieved chunks: "
        f"{result['rag_evidence']['result_count']}"
    )

    for item in result[
        "rag_evidence"
    ]["evidence"]:
        print(
            f"  - {item['source_file']} "
            f"(chunk {item['chunk_index']})"
        )

    print("\nMULTI-AGENT WORKFLOW: PASS")


if __name__ == "__main__":
    main()
