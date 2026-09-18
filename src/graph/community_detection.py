from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from src.graph.neo4j_client import get_driver


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "outputs" / "graph"
COMMUNITY_OUTPUT = OUTPUT_DIR / "communities.json"


RANDOM_STATE = 42


def load_claim_graph() -> nx.Graph:
    graph = nx.Graph()

    driver = get_driver()

    try:
        with driver.session() as session:
            records = session.run(
                """
                MATCH (c1:Claim)-[:SHARES_LOCATION_WITH]-(c2:Claim)
                RETURN c1.claim_id AS claim_1,
                       c2.claim_id AS claim_2
                """
            )

            edge_count = 0

            for record in records:
                claim_1 = record["claim_1"]
                claim_2 = record["claim_2"]

                graph.add_edge(
                    claim_1,
                    claim_2,
                )

                edge_count += 1

            print(
                f"Graph edges loaded: {edge_count:,}"
            )

    finally:
        driver.close()

    return graph


def main() -> None:
    print("=" * 70)
    print("LOUVAIN COMMUNITY DETECTION")
    print("=" * 70)

    graph = load_claim_graph()

    if graph.number_of_nodes() == 0:
        raise RuntimeError(
            "Claim graph contains no connected nodes."
        )

    print(
        f"Connected claim nodes: "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"Claim edges: "
        f"{graph.number_of_edges():,}"
    )

    communities = nx.community.louvain_communities(
        graph,
        weight=None,
        resolution=1.0,
        seed=RANDOM_STATE,
    )

    communities = sorted(
        communities,
        key=len,
        reverse=True,
    )

    community_records = []

    for community_id, members in enumerate(
        communities
    ):
        community_records.append(
            {
                "community_id": community_id,
                "size": len(members),
                "claim_ids": sorted(members),
            }
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with COMMUNITY_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "algorithm": "louvain",
                "seed": RANDOM_STATE,
                "resolution": 1.0,
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
                "community_count": len(
                    community_records
                ),
                "communities": community_records,
            },
            f,
            indent=2,
        )

    print("\nCOMMUNITY SUMMARY")
    print("-" * 70)
    print(
        f"Communities detected: "
        f"{len(community_records):,}"
    )

    print("\nLargest communities:")
    for record in community_records[:10]:
        print(
            f"Community {record['community_id']:>4}: "
            f"{record['size']:>3} claims"
        )

    print(
        f"\nSaved: {COMMUNITY_OUTPUT}"
    )

    print("\nLOUVAIN COMMUNITY DETECTION: PASS")


if __name__ == "__main__":
    main()
