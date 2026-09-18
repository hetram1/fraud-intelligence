from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "car_insurance_fraud_dataset.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EXPECTED_COLUMNS = [
    "policy_id",
    "policy_state",
    "policy_deductible",
    "policy_annual_premium",
    "insured_age",
    "insured_sex",
    "insured_education_level",
    "insured_occupation",
    "insured_hobbies",
    "incident_date",
    "incident_type",
    "collision_type",
    "incident_severity",
    "authorities_contacted",
    "incident_state",
    "incident_city",
    "incident_hour_of_the_day",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "police_report_available",
    "claim_amount",
    "total_claim_amount",
    "fraud_reported",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def main() -> None:
    print("=" * 70)
    print("CANONICAL DATA NORMALIZATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    if not RAW_FILE.exists():
        fail(f"Raw dataset does not exist: {RAW_FILE}")

    print(f"Input : {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        fail(f"Missing required columns: {missing_columns}")

    print(f"[OK] Loaded {len(df):,} raw records.")

    # ------------------------------------------------------------------
    # 2. Standardize string columns
    # ------------------------------------------------------------------
    string_columns = df.select_dtypes(include=["object", "string"]).columns

    for column in string_columns:
        df[column] = df[column].astype("string").str.strip()

    print("[OK] String fields standardized.")

    # ------------------------------------------------------------------
    # 3. Standardize dates
    # ------------------------------------------------------------------
    df["incident_date"] = pd.to_datetime(
        df["incident_date"],
        errors="coerce",
    )

    if df["incident_date"].isna().any():
        fail("Some incident_date values could not be parsed.")

    print("[OK] Dates standardized.")

    # ------------------------------------------------------------------
    # 4. Normalize fraud label
    # ------------------------------------------------------------------
    fraud_map = {
        "Y": True,
        "N": False,
    }

    df["fraud_label"] = df["fraud_reported"].map(fraud_map)

    if df["fraud_label"].isna().any():
        fail("Unexpected fraud_reported values encountered.")

    df["fraud_label"] = df["fraud_label"].astype(bool)

    # ------------------------------------------------------------------
    # 5. Generate deterministic internal claim IDs
    # ------------------------------------------------------------------
    df["claim_id"] = "CLM_" + df["policy_id"].astype(str)

    if df["claim_id"].duplicated().any():
        fail("Generated claim_id values are not unique.")

    print("[OK] Deterministic claim IDs generated.")

    # ------------------------------------------------------------------
    # 6. Build canonical policy table
    # ------------------------------------------------------------------
    policies = pd.DataFrame(
        {
            "policy_id": df["policy_id"],
            "customer_id": pd.Series(
                [pd.NA] * len(df),
                dtype="string",
            ),
            "policy_type": pd.Series(
                [pd.NA] * len(df),
                dtype="string",
            ),
            "policy_start_date": pd.Series(
                [pd.NA] * len(df),
                dtype="string",
            ),
            "policy_end_date": pd.Series(
                [pd.NA] * len(df),
                dtype="string",
            ),
            "premium": df["policy_annual_premium"],
            "deductible": df["policy_deductible"],
            "coverage_amount": pd.Series(
                [pd.NA] * len(df),
                dtype="Float64",
            ),
            "status": pd.Series(
                [pd.NA] * len(df),
                dtype="string",
            ),
        }
    )

    # ------------------------------------------------------------------
    # 7. Build canonical claim table
    # ------------------------------------------------------------------
    claims = pd.DataFrame(
        {
            "claim_id": df["claim_id"],
            "policy_id": df["policy_id"],
            "customer_id": pd.Series(
                [pd.NA] * len(df),
                dtype="string",
            ),
            # No separate claim date exists in the source.
            # We use incident_date as a documented proxy.
            "claim_date": df["incident_date"],
            "incident_date": df["incident_date"],
            "claim_type": df["incident_type"],
            "claim_description": pd.Series(
                [pd.NA] * len(df),
                dtype="string",
            ),
            "claimed_amount": df["claim_amount"],
            "approved_amount": pd.Series(
                [pd.NA] * len(df),
                dtype="Float64",
            ),
            "status": pd.Series(
                [pd.NA] * len(df),
                dtype="string",
            ),
            "fraud_label": df["fraud_label"],
        }
    )

    # Add source incident attributes required later by the ML/graph layers.
    claims["policy_state"] = df["policy_state"]
    claims["insured_age"] = df["insured_age"]
    claims["insured_sex"] = df["insured_sex"]
    claims["insured_education_level"] = df["insured_education_level"]
    claims["insured_occupation"] = df["insured_occupation"]
    claims["insured_hobbies"] = df["insured_hobbies"]

    claims["collision_type"] = df["collision_type"]
    claims["incident_severity"] = df["incident_severity"]
    claims["authorities_contacted"] = df["authorities_contacted"]

    claims["incident_state"] = df["incident_state"]
    claims["incident_city"] = df["incident_city"]
    claims["incident_hour_of_the_day"] = df[
        "incident_hour_of_the_day"
    ]

    claims["number_of_vehicles_involved"] = df[
        "number_of_vehicles_involved"
    ]

    claims["bodily_injuries"] = df["bodily_injuries"]
    claims["witnesses"] = df["witnesses"]
    claims["police_report_available"] = df["police_report_available"]

    claims["total_claim_amount"] = df["total_claim_amount"]

    # Keep the exact source dataset identity attached to every row.
    claims["source_dataset"] = (
        "ahluwaliasaksham/car-insurance-fraud-detection-dataset"
    )

    # ------------------------------------------------------------------
    # 8. Format dates for stable CSV representation
    # ------------------------------------------------------------------
    for table in (policies, claims):
        for column in [
            "policy_start_date",
            "policy_end_date",
            "claim_date",
            "incident_date",
        ]:
            if column in table.columns:
                table[column] = table[column].astype("string")

    # ------------------------------------------------------------------
    # 9. Create output directory
    # ------------------------------------------------------------------
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    policies_path = PROCESSED_DIR / "policies.csv"
    claims_path = PROCESSED_DIR / "claims.csv"

    policies.to_csv(policies_path, index=False)
    claims.to_csv(claims_path, index=False)

    # ------------------------------------------------------------------
    # 10. Generate a provenance manifest
    # ------------------------------------------------------------------
    manifest = pd.DataFrame(
        [
            {
                "source_file": str(RAW_FILE.relative_to(PROJECT_ROOT)),
                "source_sha256": sha256_file(RAW_FILE),
                "source_rows": len(df),
                "policy_rows": len(policies),
                "claim_rows": len(claims),
                "fraud_claims": int(claims["fraud_label"].sum()),
                "non_fraud_claims": int(
                    (~claims["fraud_label"]).sum()
                ),
                "output_policies": str(
                    policies_path.relative_to(PROJECT_ROOT)
                ),
                "output_claims": str(
                    claims_path.relative_to(PROJECT_ROOT)
                ),
            }
        ]
    )

    manifest_path = PROCESSED_DIR / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    # ------------------------------------------------------------------
    # 11. Sanity checks
    # ------------------------------------------------------------------
    if len(policies) != len(df):
        fail("Policy record count changed unexpectedly.")

    if len(claims) != len(df):
        fail("Claim record count changed unexpectedly.")

    if claims["claim_id"].duplicated().any():
        fail("Duplicate claim IDs exist after normalization.")

    if claims["policy_id"].duplicated().any():
        fail("Duplicate policy IDs exist after normalization.")

    print("[OK] Canonical policy table created.")
    print("[OK] Canonical claim table created.")
    print("[OK] Provenance manifest created.")

    # ------------------------------------------------------------------
    # 12. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("NORMALIZATION COMPLETE")
    print("=" * 70)

    print(f"Policies : {len(policies):,}")
    print(f"Claims   : {len(claims):,}")
    print(
        f"Fraud    : {int(claims['fraud_label'].sum()):,}"
    )
    print(
        f"Non-fraud: {int((~claims['fraud_label']).sum()):,}"
    )

    print("\nOutputs:")
    print(f"  {policies_path}")
    print(f"  {claims_path}")
    print(f"  {manifest_path}")


if __name__ == "__main__":
    main()
