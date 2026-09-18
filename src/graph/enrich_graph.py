from __future__ import annotations

from src.graph.neo4j_client import get_driver


def main() -> None:
    driver = get_driver()

    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (l:IncidentLocation)<-[:OCCURRED_AT]-(c1:Claim)
                MATCH (l)<-[:OCCURRED_AT]-(c2:Claim)
                WHERE c1.claim_id < c2.claim_id
                MERGE (c1)-[:SHARES_LOCATION_WITH]->(c2)
                RETURN count(*) AS created
                """
            ).single()

            created = result["created"]

            count_result = session.run(
                """
                RETURN count {
                    ()-[:SHARES_LOCATION_WITH]->()
                } AS relationships
                """
            ).single()

            relationships = count_result["relationships"]

            print(f"Location-sharing claim pairs created: {created:,}")
            print(f"Total SHARES_LOCATION_WITH relationships: {relationships:,}")

            top_locations = session.run(
                """
                MATCH (l:IncidentLocation)<-[:OCCURRED_AT]-(c:Claim)
                WITH l, count(c) AS claim_count
                WHERE claim_count > 1
                RETURN
                    l.state AS state,
                    l.city AS city,
                    claim_count
                ORDER BY claim_count DESC
                LIMIT 10
                """
            )

            print("\nTOP SHARED INCIDENT LOCATIONS")
            print("-" * 70)

            for record in top_locations:
                print(
                    f"{record['state']} | "
                    f"{record['city']} | "
                    f"{record['claim_count']} claims"
                )

    finally:
        driver.close()

    print("\nGRAPH ENRICHMENT: PASS")


if __name__ == "__main__":
    main()
