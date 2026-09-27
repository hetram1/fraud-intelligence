from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agents.llm_investigation_workflow import (
    build_llm_investigation,
)


app = FastAPI(
    title="Fraud Intelligence API",
    version="0.1.0",
    description="Multi-agent Graph-RAG insurance fraud investigation API.",
)


class InvestigationRequest(BaseModel):
    claim_id: str = Field(
        ...,
        min_length=1,
        description="Claim identifier to investigate.",
    )
    question: str = Field(
        ...,
        min_length=1,
        description="Investigation question.",
    )


class InvestigationResponse(BaseModel):
    question: str
    claim_id: str
    route: str
    evidence: dict[str, Any]
    report: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "fraud-intelligence-api",
    }


@app.post(
    "/investigate",
    response_model=InvestigationResponse,
)
def investigate(
    request: InvestigationRequest,
) -> InvestigationResponse:

    try:
        result = build_llm_investigation(
            question=request.question,
            claim_id=request.claim_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return InvestigationResponse(**result)
