from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "car_insurance_fraud_dataset.csv"
POLICIES_FILE = PROJECT_ROOT / "data" / "processed" / "policies.csv"
CLAIMS_FILE = PROJECT_ROOT / "data" / "processed" / "claims.csv"
MANIFEST_FILE = PROJECT_ROOT / "data" / "processed" / "manifest.csv"


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def main() -> None:
    print("=" * 70)
    print("PROCESSED DATA VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Verify all expected files exist
    # ---------------------------------------------------------------
    for path in [RAW_FILE, POLICIES_FILE, CLAIMS_FILE, MANIFEST_FILE]:
        if not path.exists():
            fail(f"Required file does not exist: {path}")

    print("[OK] All required data files exist.")

    # ---------------------------------------------------------------
    # 2. Load processed tables
    # ---------------------------------------------------------------
    policies = pd.read_csv(POLICIES_FILE)
    claims = pd.read_csv(CLAIMS_FILE)
    manifest = pd.read_csv(MANIFEST_FILE)

    print(f"[OK] Policies loaded: {len(policies):,}")
    print(f"[OK] Claims loaded:   {len(claims):,}")
    print(f"[OK] Manifest loaded: {len(manifest):,} record(s).")

    # ---------------------------------------------------------------
    # 3. Row-count preservation
    # ---------------------------------------------------------------
    raw = pd.read_csv(RAW_FILE)

    if len(policies) != len(raw):
        fail(
            f"Policy row count mismatch: "
            f"raw={len(raw):,}, processed={len(policies):,}"
        )

    if len(claims) != len(raw):
        fail(
            f"Claim row count mismatch: "
            f"raw={len(raw):,}, processed={len(claims):,}"
        )

    print("[OK] Raw-to-processed row counts preserved.")

    # ---------------------------------------------------------------
    # 4. Policy identity validation
    # ---------------------------------------------------------------
    if policies["policy_id"].isna().any():
        fail("Processed policies contain missing policy_id values.")

    duplicate_policies = int(policies["policy_id"].duplicated().sum())

    if duplicate_policies:
        fail(
            f"Processed policies contain "
            f"{duplicate_policies:,} duplicate policy IDs."
        )

    print("[OK] Processed policy IDs are unique.")

    # ---------------------------------------------------------------
    # 5. Claim identity validation
    # ---------------------------------------------------------------
    if claims["claim_id"].isna().any():
        fail("Processed claims contain missing claim_id values.")

    duplicate_claims = int(claims["claim_id"].duplicated().sum())

    if duplicate_claims:
        fail(
            f"Processed claims contain "
            f"{duplicate_claims:,} duplicate claim IDs."
        )

    print("[OK] Processed claim IDs are unique.")

    # ---------------------------------------------------------------
    # 6. Referential integrity: claim -> policy
    # ---------------------------------------------------------------
    policy_ids = set(policies["policy_id"])

    invalid_claim_policies = int(
        (~claims["policy_id"].isin(policy_ids)).sum()
    )

    if invalid_claim_policies:
        fail(
            f"{invalid_claim_policies:,} claims reference "
            f"unknown policies."
        )

    print("[OK] Claim-to-policy relationships are valid.")

    # ---------------------------------------------------------------
    # 7. Fraud-label validation
    # ---------------------------------------------------------------
    allowed_labels = {True, False}

    fraud_values = set(claims["fraud_label"].dropna().unique())

    if not fraud_values.issubset(allowed_labels):
        fail(
            f"Unexpected fraud_label values: "
            f"{sorted(fraud_values, key=str)}"
        )

    missing_fraud = int(claims["fraud_label"].isna().sum())

    if missing_fraud:
        fail(
            f"Processed claims contain "
            f"{missing_fraud:,} missing fraud labels."
        )

    fraud_count = int(claims["fraud_label"].sum())
    non_fraud_count = int((~claims["fraud_label"]).sum())

    if fraud_count != 3440:
        fail(
            f"Unexpected fraud count: expected 3,440, "
            f"found {fraud_count:,}."
        )

    if non_fraud_count != 26560:
        fail(
            f"Unexpected non-fraud count: expected 26,560, "
            f"found {non_fraud_count:,}."
        )

    print("[OK] Fraud labels and distribution are preserved.")

    # ---------------------------------------------------------------
    # 8. Date validation
    # ---------------------------------------------------------------
    claim_dates = pd.to_datetime(
        claims["claim_date"],
        errors="coerce",
    )

    incident_dates = pd.to_datetime(
        claims["incident_date"],
        errors="coerce",
    )

    if claim_dates.isna().any():
        fail("Processed claim_date contains invalid dates.")

    if incident_dates.isna().any():
        fail("Processed incident_date contains invalid dates.")

    print("[OK] Processed dates are valid.")

    # ---------------------------------------------------------------
    # 9. Manifest validation
    # ---------------------------------------------------------------
    if len(manifest) != 1:
        fail(
            f"Expected exactly one manifest record, "
            f"found {len(manifest):,}."
        )

    manifest_row = manifest.iloc[0]

    expected_manifest_values = {
        "source_rows": len(raw),
        "policy_rows": len(policies),
        "claim_rows": len(claims),
        "fraud_claims": fraud_count,
        "non_fraud_claims": non_fraud_count,
    }

    for column, expected in expected_manifest_values.items():
        actual = int(manifest_row[column])

        if actual != expected:
            fail(
                f"Manifest mismatch for {column}: "
                f"expected {expected}, found {actual}."
            )

    print("[OK] Provenance manifest is internally consistent.")

    # ---------------------------------------------------------------
    # 10. Final result
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PROCESSED DATA VALIDATION: PASS")
    print("=" * 70)

    print(f"Policies       : {len(policies):,}")
    print(f"Claims         : {len(claims):,}")
    print(f"Fraud          : {fraud_count:,}")
    print(f"Non-fraud      : {non_fraud_count:,}")
    print(f"Duplicate IDs  : 0")
    print(f"Broken links   : 0")
    print("Manifest       : consistent")


if __name__ == "__main__":
    main()
