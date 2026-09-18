from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "features"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

TARGET = "fraud_label"


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def load_split(name: str) -> pd.DataFrame:
    path = SPLIT_DIR / f"{name}.csv"

    if not path.exists():
        fail(f"Missing split file: {path}")

    df = pd.read_csv(path)

    if TARGET not in df.columns:
        fail(f"{TARGET} missing from {name} split.")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic features without learning from the dataset."""

    data = df.copy()

    # ---------------------------------------------------------------
    # Date-derived features
    # ---------------------------------------------------------------
    incident_date = pd.to_datetime(
        data["incident_date"],
        errors="coerce",
    )

    if incident_date.isna().any():
        fail("Invalid incident_date encountered during feature engineering.")

    data["incident_year"] = incident_date.dt.year
    data["incident_month"] = incident_date.dt.month
    data["incident_day_of_month"] = incident_date.dt.day
    data["incident_day_of_week"] = incident_date.dt.dayofweek
    data["incident_day_of_year"] = incident_date.dt.dayofyear
    data["incident_is_weekend"] = (
        incident_date.dt.dayofweek >= 5
    ).astype(int)

    data["incident_hour_sin"] = np.sin(
        2 * np.pi * data["incident_hour_of_the_day"] / 24.0
    )

    data["incident_hour_cos"] = np.cos(
        2 * np.pi * data["incident_hour_of_the_day"] / 24.0
    )

    # ---------------------------------------------------------------
    # Financial relationship features
    # ---------------------------------------------------------------
    premium = data["premium"] if "premium" in data.columns else None

    if premium is None:
        # Split files do not contain the canonical policy table.
        # The source feature equivalent is not available here.
        data["claim_to_total_ratio"] = (
            data["claimed_amount"]
            / data["total_claim_amount"].replace(0, np.nan)
        )
    else:
        data["claim_to_premium_ratio"] = (
            data["claimed_amount"]
            / premium.replace(0, np.nan)
        )

    data["claim_difference_from_total"] = (
        data["total_claim_amount"] - data["claimed_amount"]
    )

    # ---------------------------------------------------------------
    # Remove fields that are identifiers or canonical placeholders
    # unavailable in this source dataset.
    # ---------------------------------------------------------------
    columns_to_drop = [
        TARGET,
        "customer_id",
        "claim_description",
        "approved_amount",
        "status",
        "claim_date",
        "incident_date",
    ]

    # Do not silently fail if a field is absent.
    data = data.drop(
        columns=columns_to_drop,
        errors="ignore",
    )

    return data


def main() -> None:
    print("=" * 70)
    print("MODEL FEATURE ENGINEERING")
    print("=" * 70)

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    print(f"Raw train rows      : {len(train):,}")
    print(f"Raw validation rows : {len(validation):,}")
    print(f"Raw test rows       : {len(test):,}")

    y_train = train[TARGET].astype(bool)
    y_validation = validation[TARGET].astype(bool)
    y_test = test[TARGET].astype(bool)

    X_train = engineer_features(train)
    X_validation = engineer_features(validation)
    X_test = engineer_features(test)

    if list(X_train.columns) != list(X_validation.columns):
        fail("Train and validation feature columns differ.")

    if list(X_train.columns) != list(X_test.columns):
        fail("Train and test feature columns differ.")

    print(f"\nRaw engineered feature columns: {len(X_train.columns):,}")

    # ---------------------------------------------------------------
    # Identify feature types from the training dataframe.
    # ---------------------------------------------------------------
    numeric_columns = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    print(f"Numeric columns     : {len(numeric_columns):,}")
    print(f"Categorical columns : {len(categorical_columns):,}")

    print("\nCategorical columns:")
    for column in categorical_columns:
        print(f"  - {column}")

    print("\nNumeric columns:")
    for column in numeric_columns:
        print(f"  - {column}")

    # ---------------------------------------------------------------
    # Numeric preprocessing
    #
    # Median imputation and scaling are learned from TRAIN ONLY.
    # ---------------------------------------------------------------
    numeric_pipeline = (
        "numeric",
        __import__("sklearn").pipeline.Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        ),
        numeric_columns,
    )

    # ---------------------------------------------------------------
    # Categorical preprocessing
    #
    # min_frequency controls very rare categories, particularly the
    # high-cardinality incident_city feature.
    #
    # All vocabulary is learned from TRAIN ONLY.
    # ---------------------------------------------------------------
    categorical_pipeline = (
        "categorical",
        __import__("sklearn").pipeline.Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=10,
                        sparse_output=True,
                    ),
                ),
            ]
        ),
        categorical_columns,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            numeric_pipeline,
            categorical_pipeline,
        ],
        remainder="drop",
    )

    # ---------------------------------------------------------------
    # CRITICAL:
    # Fit ONLY on X_train.
    # ---------------------------------------------------------------
    print("\nFitting preprocessing pipeline on TRAIN only...")

    X_train_transformed = preprocessor.fit_transform(X_train)

    print("[OK] Preprocessor fitted on training data.")

    X_validation_transformed = preprocessor.transform(
        X_validation
    )

    X_test_transformed = preprocessor.transform(X_test)

    # Make sparse explicitly so storage is predictable.
    X_train_sparse = sparse.csr_matrix(X_train_transformed)
    X_validation_sparse = sparse.csr_matrix(X_validation_transformed)
    X_test_sparse = sparse.csr_matrix(X_test_transformed)

    # ---------------------------------------------------------------
    # Create output directories.
    # ---------------------------------------------------------------
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_features_path = FEATURE_DIR / "X_train.npz"
    validation_features_path = FEATURE_DIR / "X_validation.npz"
    test_features_path = FEATURE_DIR / "X_test.npz"

    sparse.save_npz(
        train_features_path,
        X_train_sparse,
    )

    sparse.save_npz(
        validation_features_path,
        X_validation_sparse,
    )

    sparse.save_npz(
        test_features_path,
        X_test_sparse,
    )

    np.save(
        FEATURE_DIR / "y_train.npy",
        y_train.to_numpy(),
    )

    np.save(
        FEATURE_DIR / "y_validation.npy",
        y_validation.to_numpy(),
    )

    np.save(
        FEATURE_DIR / "y_test.npy",
        y_test.to_numpy(),
    )

    # ---------------------------------------------------------------
    # Save the fitted preprocessing object locally.
    # It is a generated artifact and is intentionally ignored by Git.
    # ---------------------------------------------------------------
    preprocessor_path = MODEL_DIR / "preprocessor.joblib"

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    # ---------------------------------------------------------------
    # Extract model feature names.
    # ---------------------------------------------------------------
    feature_names = preprocessor.get_feature_names_out()

    with (
        FEATURE_DIR / "feature_names.json"
    ).open("w") as f:
        json.dump(
            feature_names.tolist(),
            f,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Save feature metadata.
    # ---------------------------------------------------------------
    metadata = {
        "target": TARGET,
        "numeric_input_columns": numeric_columns,
        "categorical_input_columns": categorical_columns,
        "raw_engineered_feature_count": len(X_train.columns),
        "transformed_feature_count": len(feature_names),
        "train_rows": int(X_train_sparse.shape[0]),
        "validation_rows": int(X_validation_sparse.shape[0]),
        "test_rows": int(X_test_sparse.shape[0]),
        "train_matrix_nonzero": int(X_train_sparse.nnz),
        "validation_matrix_nonzero": int(
            X_validation_sparse.nnz
        ),
        "test_matrix_nonzero": int(X_test_sparse.nnz),
        "onehot_min_frequency": 10,
        "fit_scope": "train_only",
    }

    with (
        FEATURE_DIR / "feature_metadata.json"
    ).open("w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

    print(
        f"Train matrix      : "
        f"{X_train_sparse.shape[0]:,} x "
        f"{X_train_sparse.shape[1]:,}"
    )

    print(
        f"Validation matrix : "
        f"{X_validation_sparse.shape[0]:,} x "
        f"{X_validation_sparse.shape[1]:,}"
    )

    print(
        f"Test matrix       : "
        f"{X_test_sparse.shape[0]:,} x "
        f"{X_test_sparse.shape[1]:,}"
    )

    print(
        f"Train non-zero entries      : "
        f"{X_train_sparse.nnz:,}"
    )

    print(
        f"Validation non-zero entries : "
        f"{X_validation_sparse.nnz:,}"
    )

    print(
        f"Test non-zero entries       : "
        f"{X_test_sparse.nnz:,}"
    )

    print("\nArtifacts:")
    print(f"  {train_features_path}")
    print(f"  {validation_features_path}")
    print(f"  {test_features_path}")
    print(f"  {FEATURE_DIR / 'y_train.npy'}")
    print(f"  {FEATURE_DIR / 'y_validation.npy'}")
    print(f"  {FEATURE_DIR / 'y_test.npy'}")
    print(f"  {FEATURE_DIR / 'feature_names.json'}")
    print(f"  {FEATURE_DIR / 'feature_metadata.json'}")
    print(f"  {preprocessor_path}")


if __name__ == "__main__":
    main()
