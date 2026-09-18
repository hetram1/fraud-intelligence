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


ATTRIBUTE_CONFIG = {
    "incident_type": (
        "HAS_INCIDENT_TYPE",
        "IncidentType",
        "train_same_incident_type",
    ),
    "collision_type": (
        "HAS_COLLISION_TYPE",
        "CollisionType",
        "train_same_collision_type",
    ),
    "incident_severity": (
        "HAS_SEVERITY",
        "IncidentSeverity",
        "train_same_severity",
    ),
    "incident_state": (
        "HAS_INCIDENT_STATE",
        "IncidentState",
        "train_same_incident_state",
    ),
}


def reconstruct_splits(
    claims: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train, remainder = train_test_split(
        claims,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=claims["fraud_label"],
    )

    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=remainder["fraud_label"],
    )

    return train, validation, test


def get_training_attribute_counts(
    session,
    training_claim_ids: list[str],
) -> dict[str, dict[str, int]]:

    counts = {}

    for source_column, (
        relationship,
        node_label,
        feature_name,
    ) in ATTRIBUTE_CONFIG.items():

        query = f"""
        UNWIND $training_claim_ids AS claim_id
        MATCH (c:Claim {{claim_id: claim_id}})
        MATCH (c)-[:{relationship}]->(a:{node_label})
        RETURN
            a.name AS attribute_value,
            count(*) AS training_count
        """

        records = session.run(
            query,
            training_claim_ids=training_claim_ids,
        )

        counts[source_column] = {
            str(record["attribute_value"]): int(
                record["training_count"]
            )
            for record in records
        }

        print(
            f"[OK] Training counts for "
            f"{feature_name}: "
            f"{len(counts[source_column])} categories"
        )

    return counts


def get_claim_attributes(
    session,
    claim_ids: list[str],
) -> pd.DataFrame:

    records = session.run(
        """
        UNWIND $claim_ids AS claim_id
        MATCH (c:Claim {claim_id: claim_id})

        OPTIONAL MATCH (c)-[:HAS_INCIDENT_TYPE]->(it:IncidentType)
        OPTIONAL MATCH (c)-[:HAS_COLLISION_TYPE]->(ct:CollisionType)
        OPTIONAL MATCH (c)-[:HAS_SEVERITY]->(s:IncidentSeverity)
        OPTIONAL MATCH (c)-[:HAS_INCIDENT_STATE]->(st:IncidentState)

        RETURN
            c.claim_id AS claim_id,
            it.name AS incident_type,
            ct.name AS collision_type,
            s.name AS incident_severity,
            st.name AS incident_state
        """,
        claim_ids=claim_ids,
    )

    return pd.DataFrame(
        [record.data() for record in records]
    )


def build_features_for_split(
    session,
    split_df: pd.DataFrame,
    training_counts: dict[str, dict[str, int]],
    split_name: str,
) -> pd.DataFrame:

    claim_ids = (
        split_df["claim_id"]
        .astype(str)
        .tolist()
    )

    attributes = get_claim_attributes(
        session,
        claim_ids,
    )

    expected = pd.DataFrame(
        {"claim_id": claim_ids}
    )

    attributes = expected.merge(
        attributes,
        on="claim_id",
        how="left",
        validate="one_to_one",
    )

    output = attributes[
        ["claim_id"]
    ].copy()

    for source_column, (
        _relationship,
        _node_label,
        feature_name,
    ) in ATTRIBUTE_CONFIG.items():

        output[feature_name] = (
            attributes[source_column]
            .map(
                training_counts[source_column]
            )
            .fillna(0)
            .astype(int)
        )

        # A training claim is included in its own training
        # category count. Remove itself so the feature represents
        # the number of peer training claims.
        if split_name == "train":
            output[feature_name] = (
                output[feature_name] - 1
            ).clip(lower=0)

    return output


def validate_output(
    output: pd.DataFrame,
    expected_rows: int,
    split_name: str,
) -> None:

    if len(output) != expected_rows:
        raise RuntimeError(
            f"{split_name}: expected {expected_rows} rows, "
            f"got {len(output)}"
        )

    if output["claim_id"].duplicated().any():
        raise RuntimeError(
            f"{split_name}: duplicate claim_id detected"
        )

    feature_columns = [
        value[2]
        for value in ATTRIBUTE_CONFIG.values()
    ]

    if output[feature_columns].isna().any().any():
        raise RuntimeError(
            f"{split_name}: missing graph feature values"
        )


def main() -> None:

    print("=" * 70)
    print("MULTI-RELATIONAL SPLIT-SAFE GRAPH FEATURES")
    print("=" * 70)

    if not CLAIMS_PATH.exists():
        raise FileNotFoundError(
            f"Missing claims file: {CLAIMS_PATH}"
        )

    claims = pd.read_csv(CLAIMS_PATH)

    required_columns = {
        "claim_id",
        "fraud_label",
        "incident_date",
    }

    missing = required_columns - set(claims.columns)

    if missing:
        raise RuntimeError(
            f"Claims file missing required columns: "
            f"{sorted(missing)}"
        )

    train, validation, test = reconstruct_splits(
        claims
    )

    print(f"Train rows      : {len(train):,}")
    print(f"Validation rows : {len(validation):,}")
    print(f"Test rows       : {len(test):,}")

    training_claim_ids = (
        train["claim_id"]
        .astype(str)
        .tolist()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    driver = get_driver()

    try:
        with driver.session() as session:

            print(
                "\nBuilding training-only attribute "
                "frequency maps..."
            )

            training_counts = (
                get_training_attribute_counts(
                    session,
                    training_claim_ids,
                )
            )

            splits = {
                "train": train,
                "validation": validation,
                "test": test,
            }

            for split_name, split_df in splits.items():

                output = build_features_for_split(
                    session,
                    split_df,
                    training_counts,
                    split_name,
                )

                validate_output(
                    output,
                    len(split_df),
                    split_name,
                )

                output_path = (
                    OUTPUT_DIR
                    / f"{split_name}_multirelational_graph_features.csv"
                )

                output.to_csv(
                    output_path,
                    index=False,
                )

                print(
                    f"\n[OK] {split_name:<10}"
                    f"{len(output):,} rows"
                )

                for feature in [
                    value[2]
                    for value in ATTRIBUTE_CONFIG.values()
                ]:
                    print(
                        f"     {feature:<34}"
                        f"{output[feature].min():>4}"
                        f" - {output[feature].max():<6}"
                    )

    finally:
        driver.close()

    metadata = {
        "label_free": True,
        "inductive": True,
        "training_graph_scope": "train_only",
        "features": [
            "train_same_incident_type",
            "train_same_collision_type",
            "train_same_severity",
            "train_same_incident_state",
        ],
        "random_state": RANDOM_STATE,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "method": (
            "training-only attribute frequency maps "
            "from Neo4j attribute nodes"
        ),
    }

    metadata_path = (
        OUTPUT_DIR
        / "multirelational_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(f"\nSaved metadata: {metadata_path}")
    print(
        "\nMULTI-RELATIONAL GRAPH FEATURES: PASS"
    )


if __name__ == "__main__":
    main()
