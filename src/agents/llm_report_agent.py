from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


ROOT = Path(__file__).resolve().parents[2]

load_dotenv(ROOT / ".env")


SYSTEM_PROMPT = """You are an insurance fraud investigation assistant.

Your job is to summarize the evidence supplied to you.

Rules:
1. Use only the supplied evidence.
2. Do not invent claim facts, relationships, documents, or model results.
3. Clearly distinguish model signals from factual evidence.
4. Retrieved procedural guidance is guidance, not proof that a claim is fraudulent.
5. A model score or graph relationship alone is not proof of fraud.
6. Mention missing or unavailable evidence when relevant.
7. Do not make a definitive fraud determination.
8. Produce a concise investigator-oriented report.

Structure the response as:
- Executive Summary
- Claim Facts
- Fraud Model Signal
- Graph Evidence
- Retrieved Guidance
- Investigation Considerations
- Evidence Limitations
"""


class LLMReportAgent:
    """
    LLM specialist responsible only for turning structured,
    pre-collected evidence into a grounded investigation report.
    """

    def __init__(
        self,
        model: str | None = None,
    ) -> None:

        api_key = os.getenv("OPENAI_API_KEY")
        selected_model = (
            model
            or os.getenv("OPENAI_MODEL")
        )

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured in .env"
            )

        if not selected_model:
            raise RuntimeError(
                "OPENAI_MODEL is not configured in .env"
            )

        self.llm = ChatOpenAI(
            model=selected_model,
            temperature=0,
            api_key=api_key,
        )

        self.model_name = selected_model

    def build_prompt(
        self,
        evidence: dict[str, Any],
    ) -> str:

        evidence_json = json.dumps(
            evidence,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        return (
            f"{SYSTEM_PROMPT}\n\n"
            "EVIDENCE PACKAGE:\n"
            f"{evidence_json}\n"
        )

    def generate(
        self,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:

        if not evidence:
            raise ValueError(
                "Evidence package must not be empty."
            )

        prompt = self.build_prompt(
            evidence
        )

        response = self.llm.invoke(
            prompt
        )

        content = response.content

        if not isinstance(content, str):
            content = str(content)

        return {
            "model": self.model_name,
            "report": content,
        }


def main() -> None:

    parser = argparse.ArgumentParser(
        description="LLM investigation report agent."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and inspect the prompt without calling the LLM.",
    )

    args = parser.parse_args()

    sample_evidence = {
        "claim_id": "CLM_POL100000",
        "fraud_score": {
            "fraud_risk_score": 0.7372101545333862,
            "threshold": 0.50,
            "flagged_at_threshold": True,
        },
        "graph_evidence": {
            "claim": {
                "incident_date": "2024-06-13",
                "claim_type": "Parked Car",
                "claimed_amount": 8161.36,
                "total_claim_amount": 11677.60,
                "police_report_available": "Yes",
            },
            "location": {
                "city": "Charlesville",
                "state": "MI",
            },
            "attributes": {
                "incident_type": "Parked Car",
                "collision_type": "Front",
                "severity": "Total Loss",
                "incident_state": "MI",
            },
            "relationships": {
                "shared_location_claim_count": 0,
                "shared_location_claims": [],
            },
        },
        "rag_evidence": {
            "result_count": 3,
            "evidence": [
                {
                    "source_file": "fraud_investigation_playbook.md",
                    "chunk_index": 0,
                    "text": (
                        "Investigators should verify basic claim "
                        "facts and treat unusual attributes as "
                        "investigation signals rather than proof."
                    ),
                },
                {
                    "source_file": "claim_review_sop.md",
                    "chunk_index": 0,
                    "text": (
                        "Review claim identifiers, incident "
                        "characteristics, financial amounts, "
                        "related claims, supporting evidence, "
                        "and the model score."
                    ),
                },
                {
                    "source_file": "graph_investigation_guide.md",
                    "chunk_index": 0,
                    "text": (
                        "Graph relationships provide contextual "
                        "information and are not themselves proof "
                        "of fraudulent behavior."
                    ),
                },
            ],
        },
    }

    agent = LLMReportAgent.__new__(
        LLMReportAgent
    )

    prompt = agent.build_prompt(
        sample_evidence
    )

    print("=" * 70)
    print("LLM INVESTIGATION REPORT AGENT")
    print("=" * 70)

    if args.dry_run:
        print("\nDRY RUN — NO LLM API CALL")
        print("-" * 70)
        print(prompt)

        print("\nLLM REPORT AGENT DRY RUN: PASS")
        return

    configured = {
        "OPENAI_API_KEY": bool(
            os.getenv("OPENAI_API_KEY")
        ),
        "OPENAI_MODEL": bool(
            os.getenv("OPENAI_MODEL")
        ),
    }

    print("\nConfiguration:")
    for key, value in configured.items():
        print(f"{key}: {'configured' if value else 'missing'}")

    agent = LLMReportAgent()

    result = agent.generate(
        sample_evidence
    )

    print("\nGENERATED REPORT")
    print("-" * 70)
    print(result["report"])

    print(
        "\nLLM INVESTIGATION REPORT AGENT: PASS"
    )


if __name__ == "__main__":
    main()
