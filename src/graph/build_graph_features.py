from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.graph.neo4j_client import get_driver


ROOT = Path(__file__).resolve().parents[2]

CLAIMS_PATH = ROOT / "data" / "processed" / "claims.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "graph_features"
OUTPUT_PATH = OUTPUT_DIR / "claim_graph_features.csv"
METADATA_PATH = OUTPUT_DIR / "metadata.json"


def main() -> None:
    if not CLAIMS_PATH.exists():
        raise FileNotFoundError(f"Missing claims file: {CLAIMS_PATH}")

    claims = pd.read_csv(CLAIMS_PATH, usecols=["claim_id"])

    driver = get_driver()

    try:
        with driver.session() as session:
            records = session.run(
                """
                MATCH (c:Claim)
                OPTIONAL MATCH (c)-[:OCCURRED_AT]->(l:IncidentLocation)
                OPTIONAL MATCH (l)<-[:OCCURRED_AT]-(:Claim)
                WITH c, l, count(*) - 1 AS location_claim_count
                OPTIONAL MATCH (c)-[:SHARES_LOCATION_WITH]-(neighbor)
                WITH
                    c,
                    location_claim_count,
                    count(neighbor) AS shared_location_degree
                RETURN
                    c.claim_id AS claim_id,
                    toInteger(location_claim_count) AS location_claim_count,
                    toInteger(shared_location_degree) AS shared_location_degree
                ORDER BY c.claim_id
                """
            )

            graph_rows = [record.data() for record in records]

    finally:
        driver.close()

    graph_features = pd.DataFrame(graph_rows)

    if graph_features.empty:
        raise RuntimeError("Neo4j returned no graph features")

    graph_features = claims.merge(
        graph_features,
        on="claim_id",
        how="left",
        validate="one_to_one",
    )

    graph_features["location_claim_count"] = (
        graph_features["location_claim_count"].fillna(0).astype(int)
    )

    graph_features["shared_location_degree"] = (
        graph_features["shared_location_degree"].fillna(0).astype(int)
    )

    if graph_features["claim_id"].duplicated().any():
        raise RuntimeError("Duplicate claim_id detected in graph features")

    if len(graph_features) != len(claims):
        raise RuntimeError(
            f"Row count mismatch: claims={len(claims)}, "
            f"graph_features={len(graph_features)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    graph_features.to_csv(OUTPUT_PATH, index=False)

    metadata = {
        "rows": int(len(graph_features)),
        "features": [
            "location_claim_count",
            "shared_location_degree",
        ],
        "label_free": True,
        "source": "Neo4j",
    }

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("GRAPH FEATURE EXTRACTION")
    print("-" * 70)
    print(f"Claims processed:         {len(graph_features):,}")
    print(f"Output features:          {len(metadata['features'])}")
    print(
        f"Location count range:     "
        f"{graph_features['location_claim_count'].min()} - "
        f"{graph_features['location_claim_count'].max()}"
    )
    print(
        f"Shared degree range:      "
        f"{graph_features['shared_location_degree'].min()} - "
        f"{graph_features['shared_location_degree'].max()}"
    )
    print(f"\nSaved: {OUTPUT_PATH}")

    print("\nGRAPH FEATURE EXTRACTION: PASS")


if __name__ == "__main__":
    main()
