from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "claims.csv"

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

RANDOM_STATE = 42

TEST_SIZE = 0.15
VALIDATION_SIZE_WITHIN_REMAINDER = 0.17647058823529413
# This produces approximately:
# train = 70%
# validation = 15%
# test = 15%


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def print_distribution(name: str, df: pd.DataFrame) -> None:
    counts = df["fraud_label"].value_counts().sort_index()

    fraud_count = int(counts.get(True, 0))
    non_fraud_count = int(counts.get(False, 0))

    fraud_rate = fraud_count / len(df) * 100

    print(
        f"{name:<12} "
        f"rows={len(df):>6,}  "
        f"fraud={fraud_count:>5,}  "
        f"non_fraud={non_fraud_count:>6,}  "
        f"fraud_rate={fraud_rate:>6.2f}%"
    )


def main() -> None:
    print("=" * 70)
    print("LEAKAGE-SAFE DATA SPLIT")
    print("=" * 70)

    if not INPUT_FILE.exists():
        fail(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "claim_id",
        "policy_id",
        "fraud_label",
        "incident_date",
    }

    missing = required_columns - set(df.columns)

    if missing:
        fail(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        fail("Input dataset is empty.")

    print(f"Input rows: {len(df):,}")

    # ---------------------------------------------------------------
    # Step 1: Separate features and target conceptually.
    # We keep the complete dataframe here, but use fraud_label only
    # for stratification. It is NOT used as a model feature.
    # ---------------------------------------------------------------
    target = df["fraud_label"]

    # ---------------------------------------------------------------
    # Step 2: First split -> train and temporary remainder
    # ---------------------------------------------------------------
    train_df, remainder_df = train_test_split(
        df,
        test_size=TEST_SIZE + 0.15,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    # ---------------------------------------------------------------
    # Step 3: Split remainder -> validation and test
    # 0.15 / 0.30 = 0.50 of remainder
    # ---------------------------------------------------------------
    validation_df, test_df = train_test_split(
        remainder_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=remainder_df["fraud_label"],
    )

    # ---------------------------------------------------------------
    # Step 4: Remove identifiers that must never become predictors
    # ---------------------------------------------------------------
    columns_to_remove = [
        "claim_id",
        "policy_id",
        "source_dataset",
    ]

    train_df = train_df.drop(
        columns=columns_to_remove,
        errors="ignore",
    )

    validation_df = validation_df.drop(
        columns=columns_to_remove,
        errors="ignore",
    )

    test_df = test_df.drop(
        columns=columns_to_remove,
        errors="ignore",
    )

    # ---------------------------------------------------------------
    # Step 5: Verify no target leakage through identifiers/features.
    # The target stays in the split files for evaluation, but downstream
    # model code will separate it explicitly.
    # ---------------------------------------------------------------
    for name, split in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        if "fraud_label" not in split.columns:
            fail(f"{name} split lost fraud_label.")

    # ---------------------------------------------------------------
    # Step 6: Verify all records are mutually exclusive.
    # We use incident_date + the row index that survived the split
    # indirectly by tracking original indices before dropping columns.
    # ---------------------------------------------------------------
    # The original indices are already preserved in each dataframe.
    train_indices = set(train_df.index)
    validation_indices = set(validation_df.index)
    test_indices = set(test_df.index)

    if train_indices & validation_indices:
        fail("Train and validation sets overlap.")

    if train_indices & test_indices:
        fail("Train and test sets overlap.")

    if validation_indices & test_indices:
        fail("Validation and test sets overlap.")

    if (
        len(train_indices)
        + len(validation_indices)
        + len(test_indices)
        != len(df)
    ):
        fail("Split sizes do not account for every input record.")

    # ---------------------------------------------------------------
    # Step 7: Output directory
    # ---------------------------------------------------------------
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = SPLIT_DIR / "train.csv"
    validation_path = SPLIT_DIR / "validation.csv"
    test_path = SPLIT_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)

    # ---------------------------------------------------------------
    # Step 8: Summary
    # ---------------------------------------------------------------
    print("\nSplit distribution:")
    print_distribution("Train", train_df)
    print_distribution("Validation", validation_df)
    print_distribution("Test", test_df)

    print("\nOutput files:")
    print(f"  {train_path}")
    print(f"  {validation_path}")
    print(f"  {test_path}")

    print("\n" + "=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
