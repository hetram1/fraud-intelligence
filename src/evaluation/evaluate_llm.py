from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.llm_investigation_workflow import (
    build_llm_investigation,
)


ROOT = Path(__file__).resolve().parents[2]


REQUIRED_SECTIONS = [
    "Executive Summary",
    "Claim Facts",
    "Fraud Model Signal",
    "Graph Evidence",
    "Retrieved Guidance",
    "Investigation Considerations",
    "Evidence Limitations",
]


def evaluate_report(
    result: dict[str, Any],
) -> list[str]:
    failures: list[str] = []

    report_data = result.get("report")

    if not isinstance(report_data, dict):
        failures.append("report object missing")
        return failures

    report = report_data.get("report")

    if not isinstance(report, str) or not report.strip():
        failures.append("generated report is empty")
        return failures

    claim_id = result.get("claim_id", "")

    if claim_id not in report:
        failures.append("claim_id missing from report")

    for section in REQUIRED_SECTIONS:
        if section not in report:
            failures.append(
                f"required section missing: {section}"
            )

    import re

    lowered = report.lower()

    unsafe_patterns = [
        r"\bclaim\s+(?:is|was)\s+fraudulent\b",
        r"\bthe\s+claim\s+(?:is|was)\s+fraudulent\b",
        r"\bconfirmed\s+fraud\b",
        r"\bdefinitively\s+fraudulent\b",
        r"\bdefinitive\s+fraud\s+determination\b",
    ]

    negation_markers = [
        "no ",
        "not ",
        "cannot ",
        "can't ",
        "does not ",
        "do not ",
        "did not ",
        "without ",
        "rather than ",
    ]

    sentences = re.split(r"(?<=[.!?])\s+", lowered)

    for sentence in sentences:
        for pattern in unsafe_patterns:
            match = re.search(pattern, sentence)

            if not match:
                continue

            prefix = sentence[:match.start()]

            if any(marker in prefix[-80:] for marker in negation_markers):
                continue

            failures.append(
                f"unsafe definitive statement detected: "
                f"{match.group(0)}"
            )

    if "risk score" not in lowered:
        failures.append(
            "fraud model risk score not discussed"
        )

    if "limitation" not in lowered:
        failures.append(
            "evidence limitations not discussed"
        )

    return failures


def main() -> None:
    claim_id = "CLM_POL100000"
    question = (
        "Investigate this claim for potential fraud risk."
    )

    print("=" * 70)
    print("LLM REPORT QUALITY EVALUATION")
    print("=" * 70)

    print("\nRunning Gemini-powered investigation...")
    result = build_llm_investigation(
        question=question,
        claim_id=claim_id,
    )

    failures = evaluate_report(result)

    print("\nEvaluation checks:")
    print("-" * 70)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")

        print("\nLLM REPORT EVALUATION: FAIL")
        raise RuntimeError(
            f"Report quality evaluation failed with "
            f"{len(failures)} issue(s)."
        )

    print("PASS: Report generated")
    print("PASS: Claim ID present")
    print("PASS: Required sections present")
    print("PASS: Model risk signal discussed")
    print("PASS: No definitive fraud determination")
    print("PASS: Evidence limitations discussed")

    print("\nLLM REPORT EVALUATION: PASS")


if __name__ == "__main__":
    main()
