from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.fraud_scoring_agent import FraudScoringAgent
from src.agents.graph_investigation_agent import GraphInvestigationAgent
from src.agents.rag_evidence_agent import RAGEvidenceAgent
from src.router.intent_router import IntentRouter


class RoutedInvestigationState(TypedDict, total=False):
    claim_id: str
    question: str
    route: str
    route_reason: str
    fraud_score: dict[str, Any]
    graph_evidence: dict[str, Any]
    rag_evidence: dict[str, Any]


fraud_agent = FraudScoringAgent()
graph_agent = GraphInvestigationAgent()
rag_agent = RAGEvidenceAgent(top_k=3)
router = IntentRouter()


def route_node(
    state: RoutedInvestigationState,
) -> RoutedInvestigationState:

    decision = router.route(
        state["question"]
    )

    return {
        **state,
        "route": decision.route,
        "route_reason": decision.reason,
    }


def fraud_node(
    state: RoutedInvestigationState,
) -> RoutedInvestigationState:

    return {
        **state,
        "fraud_score": fraud_agent.score(
            state["claim_id"]
        ),
    }


def graph_node(
    state: RoutedInvestigationState,
) -> RoutedInvestigationState:

    return {
        **state,
        "graph_evidence": graph_agent.investigate(
            state["claim_id"]
        ),
    }


def rag_node(
    state: RoutedInvestigationState,
) -> RoutedInvestigationState:

    return {
        **state,
        "rag_evidence": rag_agent.retrieve_evidence(
            claim_id=state["claim_id"],
            question=state["question"],
        ),
    }


def route_decision(
    state: RoutedInvestigationState,
) -> str:

    return state["route"]


def build_workflow():

    workflow = StateGraph(
        RoutedInvestigationState
    )

    workflow.add_node(
        "router",
        route_node,
    )

    workflow.add_node(
        "fraud",
        fraud_node,
    )

    workflow.add_node(
        "graph",
        graph_node,
    )

    workflow.add_node(
        "rag",
        rag_node,
    )

    workflow.add_edge(
        START,
        "router",
    )

    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "fraud": "fraud",
            "graph": "graph",
            "rag": "rag",
            "investigation": "fraud",
        },
    )

    workflow.add_edge(
        "fraud",
        END,
    )

    workflow.add_edge(
        "graph",
        END,
    )

    workflow.add_edge(
        "rag",
        END,
    )

    return workflow.compile()


def build_full_investigation_workflow():

    workflow = StateGraph(
        RoutedInvestigationState
    )

    workflow.add_node(
        "router",
        route_node,
    )

    workflow.add_node(
        "fraud",
        fraud_node,
    )

    workflow.add_node(
        "graph",
        graph_node,
    )

    workflow.add_node(
        "rag",
        rag_node,
    )

    workflow.add_edge(
        START,
        "router",
    )

    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "fraud": "fraud",
            "graph": "graph",
            "rag": "rag",
            "investigation": "fraud",
        },
    )

    workflow.add_edge(
        "fraud",
        "graph",
    )

    workflow.add_edge(
        "graph",
        "rag",
    )

    workflow.add_edge(
        "rag",
        END,
    )

    return workflow.compile()


def main() -> None:

    print("=" * 70)
    print("ROUTED MULTI-AGENT INVESTIGATION WORKFLOW")
    print("=" * 70)

    tests = [
        (
            "What is the fraud risk score for this claim?",
            "fraud",
        ),
        (
            "Which claims are related through the same location?",
            "graph",
        ),
        (
            "What is the standard procedure for reviewing "
            "a suspicious claim?",
            "rag",
        ),
        (
            "Investigate this potentially suspicious claim.",
            "investigation",
        ),
    ]

    print("\nROUTER INTEGRATION TESTS")
    print("-" * 70)

    workflow = build_workflow()

    for question, expected_route in tests:

        result = workflow.invoke(
            {
                "claim_id": "CLM_POL100000",
                "question": question,
            }
        )

        print(f"\nQuestion: {question}")
        print(f"Expected route: {expected_route}")
        print(f"Actual route:   {result['route']}")

        if result["route"] != expected_route:
            raise RuntimeError(
                f"Route mismatch: expected "
                f"{expected_route}, got "
                f"{result['route']}"
            )

    print("\n[OK] All routing tests passed.")

    print("\nFULL INVESTIGATION TEST")
    print("-" * 70)

    full_workflow = (
        build_full_investigation_workflow()
    )

    result = full_workflow.invoke(
        {
            "claim_id": "CLM_POL100000",
            "question": (
                "Investigate this potentially "
                "suspicious claim."
            ),
        }
    )

    print(
        f"Route: {result['route']}"
    )

    print(
        f"Fraud score: "
        f"{result['fraud_score']['fraud_risk_score']:.6f}"
    )

    print(
        "Graph evidence: "
        "available"
        if "graph_evidence" in result
        else "Graph evidence: missing"
    )

    print(
        f"RAG chunks: "
        f"{result['rag_evidence']['result_count']}"
    )

    print("\nROUTED MULTI-AGENT WORKFLOW: PASS")


if __name__ == "__main__":
    main()
