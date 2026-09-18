from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "car_insurance_fraud_dataset.csv"

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

NUMERIC_COLUMNS = [
    "policy_deductible",
    "policy_annual_premium",
    "insured_age",
    "incident_hour_of_the_day",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "claim_amount",
    "total_claim_amount",
]

ALLOWED_FRAUD_LABELS = {"Y", "N"}


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def main() -> None:
    print("=" * 70)
    print("RAW DATA VALIDATION")
    print("=" * 70)

    print(f"Input file: {RAW_FILE}")

    if not RAW_FILE.exists():
        fail("Raw dataset file does not exist.")

    if RAW_FILE.stat().st_size == 0:
        fail("Raw dataset file is empty.")

    print("[OK] File exists and is not empty.")

    try:
        df = pd.read_csv(RAW_FILE)
    except Exception as exc:
        fail(f"Could not read CSV: {exc}")

    print(f"[OK] CSV loaded successfully: {len(df):,} rows x {len(df.columns):,} columns.")

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------
    actual_columns = list(df.columns)

    missing_columns = [
        column for column in EXPECTED_COLUMNS if column not in actual_columns
    ]

    unexpected_columns = [
        column for column in actual_columns if column not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        fail(f"Missing required columns: {missing_columns}")

    if unexpected_columns:
        print(f"[WARN] Unexpected columns found: {unexpected_columns}")
    else:
        print("[OK] Column schema matches the expected dataset schema.")

    # ------------------------------------------------------------------
    # Identifier validation
    # ------------------------------------------------------------------
    if df["policy_id"].isna().any():
        fail("policy_id contains missing values.")

    duplicate_policy_ids = int(df["policy_id"].duplicated().sum())

    if duplicate_policy_ids:
        fail(f"Found {duplicate_policy_ids:,} duplicate policy_id values.")

    print("[OK] policy_id is present and unique.")

    # ------------------------------------------------------------------
    # Target validation
    # ------------------------------------------------------------------
    fraud_values = set(df["fraud_reported"].dropna().unique())

    invalid_fraud_labels = fraud_values - ALLOWED_FRAUD_LABELS

    if invalid_fraud_labels:
        fail(f"Invalid fraud_reported values: {sorted(invalid_fraud_labels)}")

    if df["fraud_reported"].isna().any():
        fail("fraud_reported contains missing values.")

    fraud_counts = df["fraud_reported"].value_counts()

    print("[OK] Fraud target labels are valid.")
    print("\nFraud distribution:")
    for label, count in fraud_counts.items():
        percentage = count / len(df) * 100
        print(f"  {label}: {count:,} ({percentage:.2f}%)")

    # ------------------------------------------------------------------
    # Numeric validation
    # ------------------------------------------------------------------
    print("\nNumeric column validation:")

    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(df[column], errors="coerce")
        invalid_count = int(converted.isna().sum())

        if invalid_count:
            fail(
                f"Column '{column}' contains "
                f"{invalid_count:,} non-numeric values."
            )

        print(f"  [OK] {column}")

    # ------------------------------------------------------------------
    # Date validation
    # ------------------------------------------------------------------
    parsed_dates = pd.to_datetime(df["incident_date"], errors="coerce")

    invalid_dates = int(parsed_dates.isna().sum())

    if invalid_dates:
        fail(
            f"incident_date contains "
            f"{invalid_dates:,} unparseable values."
        )

    print("[OK] incident_date values are parseable.")

    # ------------------------------------------------------------------
    # Duplicate-row validation
    # ------------------------------------------------------------------
    duplicate_rows = int(df.duplicated().sum())

    if duplicate_rows:
        fail(f"Found {duplicate_rows:,} duplicate rows.")

    print("[OK] No duplicate rows found.")

    # ------------------------------------------------------------------
    # Missing-value report
    # ------------------------------------------------------------------
    missing = df.isna().sum()
    missing = missing[missing > 0]

    print("\nMissing-value summary:")

    if missing.empty:
        print("  None")
    else:
        for column, count in missing.items():
            percentage = count / len(df) * 100
            print(f"  {column}: {count:,} ({percentage:.2f}%)")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RAW DATA VALIDATION: PASS")
    print("=" * 70)
    print(f"Rows checked        : {len(df):,}")
    print(f"Columns checked     : {len(df.columns):,}")
    print(f"Duplicate rows      : {duplicate_rows:,}")
    print(f"Duplicate policy IDs: {duplicate_policy_ids:,}")
    print(f"Fraud records       : {int((df['fraud_reported'] == 'Y').sum()):,}")
    print(f"Non-fraud records   : {int((df['fraud_reported'] == 'N').sum()):,}")


if __name__ == "__main__":
    main()
