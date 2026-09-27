from fastapi.testclient import TestClient

from src.api import app as app_module


client = TestClient(app_module.app)


def test_investigate_success(monkeypatch) -> None:
    expected = {
        "question": "Investigate this claim for potential fraud risk.",
        "claim_id": "CLM_POL100000",
        "route": "investigation",
        "evidence": {
            "claim_id": "CLM_POL100000",
            "fraud_score": {"fraud_risk_score": 0.7372},
            "graph_evidence": {"relationships": {}},
            "rag_evidence": {"result_count": 3},
        },
        "report": {
            "model": "gemini-3.5-flash-lite",
            "latency_ms": 1000.0,
            "usage": {"total_token_count": 100},
            "report": "Grounded investigation report.",
        },
    }

    def fake_investigation(
        question: str,
        claim_id: str,
    ) -> dict:
        assert question == expected["question"]
        assert claim_id == expected["claim_id"]
        return expected

    monkeypatch.setattr(
        app_module,
        "build_llm_investigation",
        fake_investigation,
    )

    response = client.post(
        "/investigate",
        json={
            "claim_id": "CLM_POL100000",
            "question": (
                "Investigate this claim "
                "for potential fraud risk."
            ),
        },
    )

    assert response.status_code == 200
    assert response.json() == expected


def test_investigate_validation_error() -> None:
    response = client.post(
        "/investigate",
        json={
            "claim_id": "",
            "question": "",
        },
    )

    assert response.status_code == 422
