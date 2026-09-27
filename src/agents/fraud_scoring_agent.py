from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix
import joblib

from src.features.build_graph_enhanced_features import engineer_features


ROOT = Path(__file__).resolve().parents[2]

CLAIMS_PATH = ROOT / "data" / "processed" / "claims.csv"

GRAPH_FEATURE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "graph_features"
)

MODEL_PATH = (
    ROOT
    / "outputs"
    / "models"
    / "xgboost_graph_enhanced.json"
)

PREPROCESSOR_PATH = (
    ROOT
    / "outputs"
    / "models"
    / "graph_enhanced_preprocessor.joblib"
)


class FraudScoringAgent:
    """
    ML specialist responsible only for producing
    a fraud-risk score from the trained model.

    This agent does not use the fraud label during inference.
    """

    def __init__(self) -> None:
        if not CLAIMS_PATH.exists():
            raise FileNotFoundError(
                f"Missing claims dataset: {CLAIMS_PATH}"
            )

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing fraud model: {MODEL_PATH}"
            )

        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(
                f"Missing graph-enhanced preprocessor: "
                f"{PREPROCESSOR_PATH}"
            )

        self.claims = pd.read_csv(CLAIMS_PATH)

        self.model = xgb.Booster()
        self.model.load_model(MODEL_PATH)

        self.preprocessor = joblib.load(
            PREPROCESSOR_PATH
        )

    def _find_claim(
        self,
        claim_id: str,
    ) -> pd.DataFrame:
        claim = self.claims[
            self.claims["claim_id"].astype(str)
            == str(claim_id)
        ].copy()

        if claim.empty:
            raise ValueError(
                f"Claim not found: {claim_id}"
            )

        if len(claim) != 1:
            raise RuntimeError(
                f"Expected one claim, found {len(claim)}"
            )

        return claim

    def _find_graph_features(
        self,
        claim_id: str,
    ) -> pd.DataFrame:

        for split_name in [
            "train",
            "validation",
            "test",
        ]:
            path = (
                GRAPH_FEATURE_DIR
                / f"{split_name}_graph_features.csv"
            )

            if not path.exists():
                continue

            graph = pd.read_csv(path)

            match = graph[
                graph["claim_id"].astype(str)
                == str(claim_id)
            ].copy()

            if len(match) == 1:
                return match

        raise ValueError(
            f"Graph features not found for claim: "
            f"{claim_id}"
        )

    def score(
        self,
        claim_id: str,
    ) -> dict[str, float | str]:

        claim = self._find_claim(
            claim_id
        )

        graph_features = self._find_graph_features(
            claim_id
        )

        features = engineer_features(
            claim,
            graph_features,
        )

        transformed = self.preprocessor.transform(
            features
        )

        matrix = csr_matrix(
            transformed
        )

        dmatrix = xgb.DMatrix(matrix)

        best_iteration = int(
            self.model.best_iteration
        )

        probability = float(
            self.model.predict(
                dmatrix,
                iteration_range=(
                    0,
                    best_iteration + 1,
                ),
            )[0]
        )

        return {
            "claim_id": str(claim_id),
            "fraud_risk_score": probability,
            "threshold": 0.50,
            "flagged_at_threshold": (
                probability >= 0.50
            ),
            "model": "xgboost_graph_enhanced",
            "best_iteration": best_iteration,
        }


def main() -> None:
    print("=" * 70)
    print("FRAUD SCORING AGENT")
    print("=" * 70)

    agent = FraudScoringAgent()

    result = agent.score(
        "CLM_POL100000"
    )

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nFRAUD SCORING AGENT: PASS")


if __name__ == "__main__":
    main()
