from __future__ import annotations

from src.graph.neo4j_client import get_driver


def main() -> None:
    print("=" * 70)
    print("SCALABLE ATTRIBUTE-NODE GRAPH ENRICHMENT")
    print("=" * 70)

    driver = get_driver()

    try:
        with driver.session() as session:

            # Remove any partially-created pairwise relationships
            # from the interrupted experiment. The location-sharing
            # relationship is intentionally preserved.
            cleanup_query = """
                MATCH ()-[r]->()
                WHERE type(r) IN [
                    'SHARES_INCIDENT_TYPE',
                    'SHARES_COLLISION_TYPE',
                    'SHARES_SEVERITY',
                    'SHARES_INCIDENT_STATE'
                ]
                DELETE r
            """

            session.run(cleanup_query).consume()
            print("[OK] Removed any partial pairwise attribute links.")

            queries = {
                "HAS_INCIDENT_TYPE": """
                    MATCH (c:Claim)
                    WHERE c.claim_type IS NOT NULL
                    MERGE (t:IncidentType {
                        name: c.claim_type
                    })
                    MERGE (c)-[:HAS_INCIDENT_TYPE]->(t)
                """,
                "HAS_COLLISION_TYPE": """
                    MATCH (c:Claim)
                    WHERE c.collision_type IS NOT NULL
                    MERGE (t:CollisionType {
                        name: c.collision_type
                    })
                    MERGE (c)-[:HAS_COLLISION_TYPE]->(t)
                """,
                "HAS_SEVERITY": """
                    MATCH (c:Claim)
                    WHERE c.incident_severity IS NOT NULL
                    MERGE (s:IncidentSeverity {
                        name: c.incident_severity
                    })
                    MERGE (c)-[:HAS_SEVERITY]->(s)
                """,
                "HAS_INCIDENT_STATE": """
                    MATCH (c:Claim)
                    WHERE c.incident_state IS NOT NULL
                    MERGE (s:IncidentState {
                        name: c.incident_state
                    })
                    MERGE (c)-[:HAS_INCIDENT_STATE]->(s)
                """,
            }

            for relationship_type, query in queries.items():
                session.run(query).consume()

                count = session.run(
                    f"""
                    MATCH ()-[:{relationship_type}]->()
                    RETURN count(*) AS count
                    """
                ).single()["count"]

                print(
                    f"[OK] {relationship_type:<28}"
                    f"{count:,} relationships"
                )

            node_counts = {
                "IncidentType": session.run(
                    "MATCH (n:IncidentType) RETURN count(n) AS count"
                ).single()["count"],
                "CollisionType": session.run(
                    "MATCH (n:CollisionType) RETURN count(n) AS count"
                ).single()["count"],
                "IncidentSeverity": session.run(
                    "MATCH (n:IncidentSeverity) RETURN count(n) AS count"
                ).single()["count"],
                "IncidentState": session.run(
                    "MATCH (n:IncidentState) RETURN count(n) AS count"
                ).single()["count"],
            }

            print("\nATTRIBUTE NODE COUNTS")
            print("-" * 70)

            for node_type, count in node_counts.items():
                print(
                    f"{node_type:<20} {count:,}"
                )

    finally:
        driver.close()

    print("\nSCALABLE ATTRIBUTE GRAPH ENRICHMENT: PASS")


if __name__ == "__main__":
    main()
