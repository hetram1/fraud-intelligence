from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]

CLAIMS_PATH = ROOT / "data" / "processed" / "claims.csv"
GRAPH_DIR = ROOT / "data" / "processed" / "graph_features"

FEATURE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "multigraph_enhanced_features"
)

MODEL_DIR = ROOT / "outputs" / "models"

TARGET = "fraud_label"
RANDOM_STATE = 42

LOCATION_COLUMNS = [
    "train_location_claim_count",
    "train_shared_location_degree",
]

ATTRIBUTE_COLUMNS = [
    "train_same_incident_type",
    "train_same_collision_type",
    "train_same_severity",
    "train_same_incident_state",
]

GRAPH_COLUMNS = LOCATION_COLUMNS + ATTRIBUTE_COLUMNS


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def reconstruct_splits(
    claims: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train, remainder = train_test_split(
        claims,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=claims[TARGET],
    )

    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=remainder[TARGET],
    )

    return train, validation, test


def load_graph_file(
    filename: str,
    required_columns: list[str],
) -> pd.DataFrame:

    path = GRAPH_DIR / filename

    if not path.exists():
        fail(f"Missing graph feature file: {path}")

    df = pd.read_csv(path)

    required = {"claim_id", *required_columns}

    missing = required - set(df.columns)

    if missing:
        fail(
            f"{filename} missing columns: "
            f"{sorted(missing)}"
        )

    if df["claim_id"].duplicated().any():
        fail(
            f"{filename} contains duplicate claim_id values."
        )

    return df[
        ["claim_id", *required_columns]
    ].copy()


def combine_graph_features(
    split_name: str,
) -> pd.DataFrame:

    location = load_graph_file(
        f"{split_name}_graph_features.csv",
        LOCATION_COLUMNS,
    )

    attributes = load_graph_file(
        f"{split_name}_multirelational_graph_features.csv",
        ATTRIBUTE_COLUMNS,
    )

    combined = location.merge(
        attributes,
        on="claim_id",
        how="inner",
        validate="one_to_one",
    )

    if combined[GRAPH_COLUMNS].isna().any().any():
        fail(
            f"{split_name}: missing graph feature values."
        )

    if combined["claim_id"].duplicated().any():
        fail(
            f"{split_name}: duplicate claim_id values."
        )

    print(
        f"[OK] {split_name:<10}"
        f" combined graph features: "
        f"{len(combined):,} rows"
    )

    return combined


def engineer_features(
    split_df: pd.DataFrame,
    graph_df: pd.DataFrame,
) -> pd.DataFrame:

    data = split_df.copy()

    data = data.merge(
        graph_df,
        on="claim_id",
        how="left",
        validate="one_to_one",
    )

    if data[GRAPH_COLUMNS].isna().any().any():
        fail("Missing graph feature values after merge.")

    incident_date = pd.to_datetime(
        data["incident_date"],
        errors="coerce",
    )

    if incident_date.isna().any():
        fail("Invalid incident_date encountered.")

    data["incident_year"] = incident_date.dt.year
    data["incident_month"] = incident_date.dt.month
    data["incident_day_of_month"] = incident_date.dt.day
    data["incident_day_of_week"] = incident_date.dt.dayofweek
    data["incident_day_of_year"] = incident_date.dt.dayofyear

    data["incident_is_weekend"] = (
        incident_date.dt.dayofweek >= 5
    ).astype(int)

    data["incident_hour_sin"] = np.sin(
        2
        * np.pi
        * data["incident_hour_of_the_day"]
        / 24.0
    )

    data["incident_hour_cos"] = np.cos(
        2
        * np.pi
        * data["incident_hour_of_the_day"]
        / 24.0
    )

    data["claim_to_total_ratio"] = (
        data["claimed_amount"]
        / data["total_claim_amount"].replace(0, np.nan)
    )

    data["claim_difference_from_total"] = (
        data["total_claim_amount"]
        - data["claimed_amount"]
    )

    columns_to_drop = [
        TARGET,
        "claim_id",
        "policy_id",
        "source_dataset",
        "customer_id",
        "claim_description",
        "approved_amount",
        "status",
        "claim_date",
        "incident_date",
    ]

    return data.drop(
        columns=columns_to_drop,
        errors="ignore",
    )


def main() -> None:

    print("=" * 70)
    print("FULL MULTI-RELATIONAL GRAPH-ENHANCED FEATURES")
    print("=" * 70)

    if not CLAIMS_PATH.exists():
        fail(f"Missing claims file: {CLAIMS_PATH}")

    claims = pd.read_csv(CLAIMS_PATH)

    required_claim_columns = {
        "claim_id",
        TARGET,
        "incident_date",
    }

    missing = (
        required_claim_columns
        - set(claims.columns)
    )

    if missing:
        fail(
            f"Claims file missing required columns: "
            f"{sorted(missing)}"
        )

    train, validation, test = reconstruct_splits(
        claims
    )

    print(f"Train rows      : {len(train):,}")
    print(f"Validation rows : {len(validation):,}")
    print(f"Test rows       : {len(test):,}")

    graph_data = {
        "train": combine_graph_features("train"),
        "validation": combine_graph_features(
            "validation"
        ),
        "test": combine_graph_features("test"),
    }

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for name in splits:

        expected_ids = set(
            splits[name]["claim_id"].astype(str)
        )

        actual_ids = set(
            graph_data[name]["claim_id"].astype(str)
        )

        if expected_ids != actual_ids:
            fail(
                f"{name}: graph claim IDs do not exactly "
                f"match the split."
            )

        if len(graph_data[name]) != len(
            splits[name]
        ):
            fail(
                f"{name}: graph feature row count mismatch."
            )

        print(
            f"[OK] {name:<10}"
            f" graph IDs exactly match split."
        )

    y_train = train[TARGET].astype(bool)
    y_validation = validation[TARGET].astype(bool)
    y_test = test[TARGET].astype(bool)

    X_train = engineer_features(
        train,
        graph_data["train"],
    )

    X_validation = engineer_features(
        validation,
        graph_data["validation"],
    )

    X_test = engineer_features(
        test,
        graph_data["test"],
    )

    if list(X_train.columns) != list(
        X_validation.columns
    ):
        fail("Train and validation feature columns differ.")

    if list(X_train.columns) != list(
        X_test.columns
    ):
        fail("Train and test feature columns differ.")

    print(
        f"\nRaw feature columns: "
        f"{len(X_train.columns):,}"
    )

    numeric_columns = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    print(
        f"Numeric columns     : "
        f"{len(numeric_columns):,}"
    )

    print(
        f"Categorical columns : "
        f"{len(categorical_columns):,}"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown=(
                                    "infrequent_if_exist"
                                ),
                                min_frequency=10,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    print(
        "\nFitting preprocessing on TRAIN only..."
    )

    X_train_transformed = (
        preprocessor.fit_transform(X_train)
    )

    X_validation_transformed = (
        preprocessor.transform(X_validation)
    )

    X_test_transformed = (
        preprocessor.transform(X_test)
    )

    X_train_sparse = sparse.csr_matrix(
        X_train_transformed
    )

    X_validation_sparse = sparse.csr_matrix(
        X_validation_transformed
    )

    X_test_sparse = sparse.csr_matrix(
        X_test_transformed
    )

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sparse.save_npz(
        FEATURE_DIR / "X_train.npz",
        X_train_sparse,
    )

    sparse.save_npz(
        FEATURE_DIR / "X_validation.npz",
        X_validation_sparse,
    )

    sparse.save_npz(
        FEATURE_DIR / "X_test.npz",
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

    joblib.dump(
        preprocessor,
        MODEL_DIR
        / "multigraph_enhanced_preprocessor.joblib",
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    with (
        FEATURE_DIR / "feature_names.json"
    ).open("w") as f:
        json.dump(
            feature_names.tolist(),
            f,
            indent=2,
        )

    metadata = {
        "target": TARGET,
        "graph_features": GRAPH_COLUMNS,
        "graph_scope": "train_only",
        "label_free_graph_features": True,
        "raw_engineered_feature_count": len(
            X_train.columns
        ),
        "transformed_feature_count": len(
            feature_names
        ),
        "train_rows": int(
            X_train_sparse.shape[0]
        ),
        "validation_rows": int(
            X_validation_sparse.shape[0]
        ),
        "test_rows": int(
            X_test_sparse.shape[0]
        ),
        "train_matrix_nonzero": int(
            X_train_sparse.nnz
        ),
        "validation_matrix_nonzero": int(
            X_validation_sparse.nnz
        ),
        "test_matrix_nonzero": int(
            X_test_sparse.nnz
        ),
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

    print("\n" + "=" * 70)
    print("MULTIGRAPH FEATURE ENGINEERING COMPLETE")
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

    print("\nMULTIGRAPH FEATURES: PASS")


if __name__ == "__main__":
    main()
