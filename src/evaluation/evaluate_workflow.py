from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agents.routed_investigation_workflow import (
    build_workflow,
    build_full_investigation_workflow,
)


ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = ROOT / "data" / "evaluation" / "investigation_eval.jsonl"


def main() -> None:
    specialist_workflow = build_workflow()
    full_workflow = build_full_investigation_workflow()

    total = 0
    route_pass = 0
    evidence_pass = 0

    print("=" * 70)
    print("INVESTIGATION WORKFLOW EVALUATION")
    print("=" * 70)

    for line in EVAL_PATH.read_text().splitlines():
        if not line.strip():
            continue

        case: dict[str, Any] = json.loads(line)

        workflow = (
            full_workflow
            if case["expected_route"] == "investigation"
            else specialist_workflow
        )

        result = workflow.invoke(
            {
                "claim_id": case["claim_id"],
                "question": case["question"],
            }
        )

        total += 1

        expected_route = case["expected_route"]
        actual_route = result["route"]

        route_ok = actual_route == expected_route

        expected_evidence = case["expected_evidence"]
        missing_evidence = [
            key
            for key in expected_evidence
            if key not in result
        ]

        evidence_ok = not missing_evidence

        if route_ok:
            route_pass += 1

        if evidence_ok:
            evidence_pass += 1

        print(f"\n{case['case_id']}")
        print(f"  Expected route: {expected_route}")
        print(f"  Actual route:   {actual_route}")
        print(f"  Route:          {'PASS' if route_ok else 'FAIL'}")
        print(
            f"  Evidence:       "
            f"{'PASS' if evidence_ok else 'FAIL'}"
        )

        if missing_evidence:
            print(
                f"  Missing evidence: {missing_evidence}"
            )

    route_accuracy = route_pass / total if total else 0.0
    evidence_accuracy = (
        evidence_pass / total if total else 0.0
    )

    print("\n" + "-" * 70)
    print(f"Cases:             {total}")
    print(
        f"Route accuracy:    "
        f"{route_pass}/{total} ({route_accuracy:.2%})"
    )
    print(
        f"Evidence accuracy: "
        f"{evidence_pass}/{total} ({evidence_accuracy:.2%})"
    )

    if route_pass != total or evidence_pass != total:
        raise RuntimeError(
            "Workflow evaluation failed."
        )

    print("\nWORKFLOW EVALUATION: PASS")


if __name__ == "__main__":
    main()
