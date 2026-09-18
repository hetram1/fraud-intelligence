from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.graph.neo4j_client import get_driver


ROOT = Path(__file__).resolve().parents[2]

POLICIES_PATH = ROOT / "data" / "processed" / "policies.csv"
CLAIMS_PATH = ROOT / "data" / "processed" / "claims.csv"


def clean_value(value):
    if pd.isna(value):
        return None
    return value


def main() -> None:
    if not POLICIES_PATH.exists():
        raise FileNotFoundError(f"Missing policies file: {POLICIES_PATH}")

    if not CLAIMS_PATH.exists():
        raise FileNotFoundError(f"Missing claims file: {CLAIMS_PATH}")

    policies = pd.read_csv(POLICIES_PATH)
    claims = pd.read_csv(CLAIMS_PATH)

    print(f"Policies loaded: {len(policies):,}")
    print(f"Claims loaded:   {len(claims):,}")

    policy_records = []
    for row in policies.itertuples(index=False):
        policy_records.append(
            {
                "policy_id": row.policy_id,
                "policy_type": clean_value(row.policy_type),
                "policy_start": clean_value(row.policy_start_date),
                "policy_end": clean_value(row.policy_end_date),
                "premium": clean_value(row.premium),
                "deductible": clean_value(row.deductible),
                "coverage_amount": clean_value(row.coverage_amount),
                "status": clean_value(row.status),
            }
        )

    claim_records = []

    for row in claims.itertuples(index=False):
        incident_city = clean_value(row.incident_city)
        incident_state = clean_value(row.incident_state)

        location_key = None

        if incident_city is not None or incident_state is not None:
            location_key = "|".join(
                [
                    str(incident_state or ""),
                    str(incident_city or ""),
                ]
            )

        claim_records.append(
            {
                "claim_id": row.claim_id,
                "policy_id": row.policy_id,
                "claim_date": clean_value(row.claim_date),
                "claim_type": clean_value(row.claim_type),
                "description": clean_value(row.claim_description),
                "claimed_amount": clean_value(row.claimed_amount),
                "approved_amount": clean_value(row.approved_amount),
                "status": clean_value(row.status),
                "fraud_label": clean_value(row.fraud_label),
                "incident_date": clean_value(row.incident_date),
                "incident_type": clean_value(row.claim_type),
                "collision_type": clean_value(row.collision_type),
                "incident_severity": clean_value(row.incident_severity),
                "authorities_contacted": clean_value(row.authorities_contacted),
                "incident_state": incident_state,
                "incident_city": incident_city,
                "incident_hour_of_day": clean_value(row.incident_hour_of_the_day),
                "number_of_vehicles_involved": clean_value(
                    row.number_of_vehicles_involved
                ),
                "bodily_injuries": clean_value(row.bodily_injuries),
                "witnesses": clean_value(row.witnesses),
                "police_report_available": clean_value(
                    row.police_report_available
                ),
                "claim_amount_source": clean_value(row.claimed_amount),
                "total_claim_amount": clean_value(row.total_claim_amount),
                "location_key": location_key,
            }
        )

    driver = get_driver()

    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n").consume()
            print("[OK] Existing graph cleared.")

            session.run(
                """
                UNWIND $policies AS policy
                MERGE (p:Policy {policy_id: policy.policy_id})
                SET
                    p.policy_type = policy.policy_type,
                    p.policy_start = policy.policy_start,
                    p.policy_end = policy.policy_end,
                    p.premium = policy.premium,
                    p.deductible = policy.deductible,
                    p.coverage_amount = policy.coverage_amount,
                    p.status = policy.status
                """,
                policies=policy_records,
            ).consume()

            print(f"[OK] Loaded {len(policy_records):,} Policy nodes.")

            session.run(
                """
                UNWIND $claims AS claim
                MATCH (p:Policy {policy_id: claim.policy_id})
                MERGE (c:Claim {claim_id: claim.claim_id})
                SET
                    c.claim_date = claim.claim_date,
                    c.claim_type = claim.claim_type,
                    c.description = claim.description,
                    c.claimed_amount = claim.claimed_amount,
                    c.approved_amount = claim.approved_amount,
                    c.status = claim.status,
                    c.fraud_label = claim.fraud_label,
                    c.incident_date = claim.incident_date,
                    c.incident_type = claim.incident_type,
                    c.collision_type = claim.collision_type,
                    c.incident_severity = claim.incident_severity,
                    c.authorities_contacted = claim.authorities_contacted,
                    c.incident_state = claim.incident_state,
                    c.incident_city = claim.incident_city,
                    c.incident_hour_of_day = claim.incident_hour_of_day,
                    c.number_of_vehicles_involved =
                        claim.number_of_vehicles_involved,
                    c.bodily_injuries = claim.bodily_injuries,
                    c.witnesses = claim.witnesses,
                    c.police_report_available =
                        claim.police_report_available,
                    c.claim_amount_source = claim.claim_amount_source,
                    c.total_claim_amount = claim.total_claim_amount
                MERGE (p)-[:HAS_CLAIM]->(c)
                WITH c, claim
                WHERE claim.location_key IS NOT NULL
                MERGE (l:IncidentLocation {
                    location_key: claim.location_key
                })
                SET
                    l.city = claim.incident_city,
                    l.state = claim.incident_state
                MERGE (c)-[:OCCURRED_AT]->(l)
                """,
                claims=claim_records,
            ).consume()

            print(f"[OK] Loaded {len(claim_records):,} Claim nodes.")

            counts = session.run(
                """
                RETURN
                    count { (p:Policy) } AS policies,
                    count { (c:Claim) } AS claims,
                    count { (l:IncidentLocation) } AS locations,
                    count { ()-[r:HAS_CLAIM]->() } AS policy_claim_edges,
                    count { ()-[r:OCCURRED_AT]->() } AS claim_location_edges
                """
            ).single()

            print("\nGRAPH COUNTS")
            print("-" * 60)
            print(f"Policy nodes:              {counts['policies']:,}")
            print(f"Claim nodes:               {counts['claims']:,}")
            print(f"IncidentLocation nodes:    {counts['locations']:,}")
            print(f"HAS_CLAIM relationships:   {counts['policy_claim_edges']:,}")
            print(f"OCCURRED_AT relationships: {counts['claim_location_edges']:,}")

    finally:
        driver.close()

    print("\nGRAPH INGESTION: PASS")


if __name__ == "__main__":
    main()
