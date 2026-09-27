from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str


class IntentRouter:
    """
    Lightweight, deterministic router for investigation requests.

    Routes:
      - fraud       -> model scoring
      - graph       -> graph investigation
      - rag         -> knowledge retrieval
      - investigation -> full multi-agent workflow
    """

    FRAUD_KEYWORDS = {
        "fraud",
        "risk",
        "score",
        "probability",
        "predict",
        "prediction",
        "model",
        "flag",
        "threshold",
    }

    GRAPH_KEYWORDS = {
        "graph",
        "relationship",
        "related",
        "connected",
        "connection",
        "location",
        "neighbor",
        "community",
        "network",
        "claims linked",
    }

    RAG_KEYWORDS = {
        "policy",
        "procedure",
        "guideline",
        "guidance",
        "sop",
        "standard",
        "how should",
        "how do i",
        "review process",
        "investigation process",
    }

    INVESTIGATION_KEYWORDS = {
        "investigate",
        "investigation",
        "case investigation",
        "full investigation",
        "full review",
        "complete investigation",
        "analyze this claim",
        "assess this claim",
    }

    def route(self, question: str) -> RouteDecision:
        question = question.strip().lower()

        if not question:
            raise ValueError(
                "Question must not be empty."
            )

        investigation_hits = self._count_hits(
            question,
            self.INVESTIGATION_KEYWORDS,
        )

        fraud_hits = self._count_hits(
            question,
            self.FRAUD_KEYWORDS,
        )

        graph_hits = self._count_hits(
            question,
            self.GRAPH_KEYWORDS,
        )

        rag_hits = self._count_hits(
            question,
            self.RAG_KEYWORDS,
        )

        if investigation_hits > 0:
            return RouteDecision(
                route="investigation",
                reason=(
                    "Question requests a case-level "
                    "or claim-level investigation."
                ),
            )

        scores = {
            "fraud": fraud_hits,
            "graph": graph_hits,
            "rag": rag_hits,
        }

        best_route = max(
            scores,
            key=scores.get,
        )

        best_score = scores[best_route]

        if best_score == 0:
            return RouteDecision(
                route="investigation",
                reason=(
                    "No specialist intent detected; "
                    "defaulting to full investigation."
                ),
            )

        return RouteDecision(
            route=best_route,
            reason=(
                f"Detected {best_score} "
                f"{best_route}-specific keyword match(es)."
            ),
        )

    @staticmethod
    def _count_hits(
        question: str,
        keywords: set[str],
    ) -> int:
        return sum(
            1
            for keyword in keywords
            if keyword in question
        )


def main() -> None:
    print("=" * 70)
    print("INVESTIGATION INTENT ROUTER")
    print("=" * 70)

    router = IntentRouter()

    test_questions = [
        (
            "What is the fraud risk score for this claim?"
        ),
        (
            "Which claims are related through "
            "the same location?"
        ),
        (
            "What is the standard procedure for "
            "reviewing a suspicious claim?"
        ),
        (
            "Investigate this potentially suspicious claim."
        ),
        (
            "Analyze this claim and provide a complete "
            "investigation."
        ),
    ]

    for question in test_questions:
        decision = router.route(question)

        print("\nQuestion:")
        print(f"  {question}")
        print(f"Route:  {decision.route}")
        print(f"Reason: {decision.reason}")

    print("\nINTENT ROUTER: PASS")


if __name__ == "__main__":
    main()
