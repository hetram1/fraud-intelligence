from __future__ import annotations

from typing import Any

from src.graph.neo4j_client import get_driver


class GraphInvestigationAgent:
    """
    Graph specialist responsible for retrieving
    relationship-based evidence from Neo4j.
    """

    def investigate(
        self,
        claim_id: str,
    ) -> dict[str, Any]:

        driver = get_driver()

        try:
            with driver.session() as session:
                record = session.run(
                    """
                    MATCH (c:Claim {claim_id: $claim_id})

                    OPTIONAL MATCH
                        (p:Policy)-[:HAS_CLAIM]->(c)

                    OPTIONAL MATCH
                        (c)-[:OCCURRED_AT]->(location)

                    OPTIONAL MATCH
                        (c)-[:SHARES_LOCATION_WITH]-
                        (neighbor:Claim)

                    OPTIONAL MATCH
                        (c)-[:HAS_INCIDENT_TYPE]->
                        (incident_type:IncidentType)

                    OPTIONAL MATCH
                        (c)-[:HAS_COLLISION_TYPE]->
                        (collision_type:CollisionType)

                    OPTIONAL MATCH
                        (c)-[:HAS_SEVERITY]->
                        (severity:IncidentSeverity)

                    OPTIONAL MATCH
                        (c)-[:HAS_INCIDENT_STATE]->
                        (incident_state:IncidentState)

                    RETURN
                        c.claim_id AS claim_id,
                        c.incident_date AS incident_date,
                        c.claim_type AS claim_type,
                        c.claimed_amount AS claimed_amount,
                        c.total_claim_amount AS total_claim_amount,
                        c.police_report_available
                            AS police_report_available,

                        p.policy_id AS policy_id,
                        p.premium AS policy_premium,
                        p.deductible AS policy_deductible,

                        location.city AS incident_city,
                        location.state AS incident_state,

                        collect(
                            DISTINCT neighbor.claim_id
                        )[0..10] AS related_claim_ids,

                        incident_type.name
                            AS incident_type,

                        collision_type.name
                            AS collision_type,

                        severity.name
                            AS severity,

                        incident_state.name
                            AS incident_state_node
                    """,
                    claim_id=claim_id,
                ).single()

                if record is None:
                    raise ValueError(
                        f"Claim not found: {claim_id}"
                    )

                related_claims = (
                    record["related_claim_ids"]
                )

                return {
                    "claim_id": record["claim_id"],
                    "claim": {
                        "claim_id": record["claim_id"],
                        "incident_date": record[
                            "incident_date"
                        ],
                        "claim_type": record[
                            "claim_type"
                        ],
                        "claimed_amount": record[
                            "claimed_amount"
                        ],
                        "total_claim_amount": record[
                            "total_claim_amount"
                        ],
                        "police_report_available": record[
                            "police_report_available"
                        ],
                    },
                    "policy": {
                        "policy_id": record["policy_id"],
                        "premium": record[
                            "policy_premium"
                        ],
                        "deductible": record[
                            "policy_deductible"
                        ],
                    },
                    "location": {
                        "city": record[
                            "incident_city"
                        ],
                        "state": record[
                            "incident_state"
                        ],
                    },
                    "attributes": {
                        "incident_type": record[
                            "incident_type"
                        ],
                        "collision_type": record[
                            "collision_type"
                        ],
                        "severity": record[
                            "severity"
                        ],
                        "incident_state": record[
                            "incident_state_node"
                        ],
                    },
                    "relationships": {
                        "shared_location_claims": (
                            related_claims
                        ),
                        "shared_location_claim_count": (
                            len(related_claims)
                        ),
                    },
                }

        finally:
            driver.close()


def main() -> None:
    print("=" * 70)
    print("GRAPH INVESTIGATION AGENT")
    print("=" * 70)

    agent = GraphInvestigationAgent()

    result = agent.investigate(
        "CLM_POL100000"
    )

    print("\nGRAPH EVIDENCE")
    print("-" * 70)

    print(f"Claim: {result['claim_id']}")

    print("\nPolicy:")
    for key, value in result["policy"].items():
        print(f"  {key}: {value}")

    print("\nLocation:")
    for key, value in result["location"].items():
        print(f"  {key}: {value}")

    print("\nAttributes:")
    for key, value in result["attributes"].items():
        print(f"  {key}: {value}")

    print("\nRelationships:")
    for key, value in result["relationships"].items():
        print(f"  {key}: {value}")

    print("\nGRAPH INVESTIGATION AGENT: PASS")


if __name__ == "__main__":
    main()
