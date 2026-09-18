from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.graph.neo4j_client import get_driver


ROOT = Path(__file__).resolve().parents[2]

CLAIMS_PATH = ROOT / "data" / "processed" / "claims.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "graph_features"


RANDOM_STATE = 42
TEST_SIZE = 0.30


def reconstruct_splits(
    claims: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, remainder_df = train_test_split(
        claims,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=claims["fraud_label"],
    )

    validation_df, test_df = train_test_split(
        remainder_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=remainder_df["fraud_label"],
    )

    return train_df, validation_df, test_df


def get_location_features(
    session,
    claim_ids: list[str],
    training_claim_ids: list[str],
) -> list[dict]:
    result = session.run(
        """
        UNWIND $claim_ids AS claim_id
        MATCH (c:Claim {claim_id: claim_id})
        OPTIONAL MATCH (c)-[:OCCURRED_AT]->(l:IncidentLocation)
        OPTIONAL MATCH (train_claim:Claim)-[:OCCURRED_AT]->(l)
        WHERE train_claim.claim_id IN $training_claim_ids
        WITH
            c,
            count(train_claim) AS train_location_claim_count
        RETURN
            c.claim_id AS claim_id,
            toInteger(train_location_claim_count) AS train_location_claim_count
        ORDER BY c.claim_id
        """,
        claim_ids=claim_ids,
        training_claim_ids=training_claim_ids,
    )

    return [record.data() for record in result]


def main() -> None:
    if not CLAIMS_PATH.exists():
        raise FileNotFoundError(f"Missing claims file: {CLAIMS_PATH}")

    claims = pd.read_csv(CLAIMS_PATH)

    required_columns = {
        "claim_id",
        "fraud_label",
        "incident_date",
    }

    missing = required_columns - set(claims.columns)

    if missing:
        raise RuntimeError(
            f"Missing required claim columns: {sorted(missing)}"
        )

    train_df, validation_df, test_df = reconstruct_splits(claims)

    print("RECONSTRUCTED SPLITS")
    print("-" * 70)
    print(f"Train:      {len(train_df):,}")
    print(f"Validation: {len(validation_df):,}")
    print(f"Test:       {len(test_df):,}")

    training_claim_ids = train_df["claim_id"].astype(str).tolist()

    driver = get_driver()

    try:
        with driver.session() as session:
            split_frames = {
                "train": train_df,
                "validation": validation_df,
                "test": test_df,
            }

            for split_name, split_df in split_frames.items():
                claim_ids = split_df["claim_id"].astype(str).tolist()

                rows = get_location_features(
                    session,
                    claim_ids,
                    training_claim_ids,
                )

                features = pd.DataFrame(rows)

                expected = pd.DataFrame(
                    {"claim_id": claim_ids}
                )

                features = expected.merge(
                    features,
                    on="claim_id",
                    how="left",
                    validate="one_to_one",
                )

                features["train_location_claim_count"] = (
                    features["train_location_claim_count"]
                    .fillna(0)
                    .astype(int)
                )

                if split_name == "train":
                    features["train_shared_location_degree"] = (
                        features["train_location_claim_count"] - 1
                    ).clip(lower=0)
                else:
                    features["train_shared_location_degree"] = (
                        features["train_location_claim_count"]
                    )

                output_path = (
                    OUTPUT_DIR
                    / f"{split_name}_graph_features.csv"
                )

                features.to_csv(output_path, index=False)

                print(
                    f"[OK] {split_name:<10} "
                    f"{len(features):,} rows → {output_path}"
                )

                if features["claim_id"].duplicated().any():
                    raise RuntimeError(
                        f"Duplicate claim_id in {split_name} features"
                    )

                if len(features) != len(split_df):
                    raise RuntimeError(
                        f"Row count mismatch for {split_name}"
                    )

    finally:
        driver.close()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "label_free": True,
        "inductive": True,
        "training_graph_scope": "train_only",
        "validation_graph_scope": "train_only",
        "test_graph_scope": "train_only",
        "features": [
            "train_location_claim_count",
            "train_shared_location_degree",
        ],
        "random_state": RANDOM_STATE,
        "split": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "test_rows": int(len(test_df)),
        },
    }

    metadata_path = OUTPUT_DIR / "split_safe_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata: {metadata_path}")
    print("\nSPLIT-SAFE GRAPH FEATURES: PASS")


if __name__ == "__main__":
    main()
