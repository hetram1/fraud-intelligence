CREATE CONSTRAINT policy_id_unique IF NOT EXISTS
FOR (p:Policy)
REQUIRE p.policy_id IS UNIQUE;

CREATE CONSTRAINT claim_id_unique IF NOT EXISTS
FOR (c:Claim)
REQUIRE c.claim_id IS UNIQUE;

CREATE CONSTRAINT incident_location_key_unique IF NOT EXISTS
FOR (l:IncidentLocation)
REQUIRE l.location_key IS UNIQUE;

CREATE INDEX claim_fraud_label IF NOT EXISTS
FOR (c:Claim)
ON (c.fraud_label);

CREATE INDEX claim_incident_date IF NOT EXISTS
FOR (c:Claim)
ON (c.incident_date);
