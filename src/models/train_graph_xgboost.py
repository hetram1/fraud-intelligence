from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy import sparse
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph_enhanced_features"
)

MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"

RANDOM_STATE = 42


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def load_matrix(name: str) -> sparse.csr_matrix:
    path = FEATURE_DIR / f"X_{name}.npz"

    if not path.exists():
        fail(f"Feature matrix not found: {path}")

    return sparse.load_npz(path).tocsr()


def load_target(name: str) -> np.ndarray:
    path = FEATURE_DIR / f"y_{name}.npy"

    if not path.exists():
        fail(f"Target file not found: {path}")

    return np.load(path).astype(bool)


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:

    predictions = probabilities >= 0.5

    metrics = {
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "precision_at_0_5": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall_at_0_5": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1_at_0_5": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
    }

    return metrics


def main() -> None:
    print("=" * 70)
    print("XGBOOST GRAPH-ENHANCED FRAUD MODEL")
    print("=" * 70)

    X_train = load_matrix("train")
    X_validation = load_matrix("validation")

    y_train = load_target("train")
    y_validation = load_target("validation")

    print(f"Train matrix      : {X_train.shape}")
    print(f"Validation matrix : {X_validation.shape}")

    positive_count = int(y_train.sum())
    negative_count = int((~y_train).sum())

    if positive_count == 0:
        fail("Training set contains no fraud examples.")

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(
        f"\nTrain fraud count     : "
        f"{positive_count:,}"
    )
    print(
        f"Train non-fraud count : "
        f"{negative_count:,}"
    )
    print(
        f"Scale positive weight : "
        f"{scale_pos_weight:.6f}"
    )

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train.astype(np.float32),
    )

    dvalidation = xgb.DMatrix(
        X_validation,
        label=y_validation.astype(np.float32),
    )

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "seed": RANDOM_STATE,
    }

    print("\nTraining XGBoost...")
    print("Parameters:")

    for key, value in params.items():
        print(f"  {key}: {value}")

    evaluation_history: dict[
        str,
        dict[str, list[float]],
    ] = {}

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[
            (dtrain, "train"),
            (dvalidation, "validation"),
        ],
        evals_result=evaluation_history,
        early_stopping_rounds=50,
        verbose_eval=50,
    )

    print("\nTraining complete.")

    best_iteration = model.best_iteration

    print(f"Best iteration: {best_iteration}")

    validation_probabilities = model.predict(
        dvalidation,
        iteration_range=(
            0,
            best_iteration + 1
            if best_iteration is not None
            else 1000,
        ),
    )

    validation_metrics = evaluate(
        y_validation,
        validation_probabilities,
    )

    print("\nValidation metrics:")
    print(
        f"  PR-AUC          : "
        f"{validation_metrics['pr_auc']:.6f}"
    )
    print(
        f"  ROC-AUC         : "
        f"{validation_metrics['roc_auc']:.6f}"
    )
    print(
        f"  Precision @ 0.5 : "
        f"{validation_metrics['precision_at_0_5']:.6f}"
    )
    print(
        f"  Recall @ 0.5    : "
        f"{validation_metrics['recall_at_0_5']:.6f}"
    )
    print(
        f"  F1 @ 0.5        : "
        f"{validation_metrics['f1_at_0_5']:.6f}"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / "xgboost_graph_enhanced.json"
    )

    model.save_model(model_path)

    metrics = {
        "model": "xgboost_graph_enhanced",
        "random_state": RANDOM_STATE,
        "best_iteration": (
            int(best_iteration)
            if best_iteration is not None
            else None
        ),
        "scale_pos_weight": float(
            scale_pos_weight
        ),
        "training_rows": int(
            X_train.shape[0]
        ),
        "validation_rows": int(
            X_validation.shape[0]
        ),
        "feature_count": int(
            X_train.shape[1]
        ),
        "graph_features": [
            "train_location_claim_count",
            "train_shared_location_degree",
        ],
        "graph_scope": "train_only",
        "validation_metrics": validation_metrics,
        "evaluation_history": evaluation_history,
    }

    metrics_path = (
        EVALUATION_DIR
        / "xgboost_graph_enhanced_validation.json"
    )

    with metrics_path.open("w") as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    print("\nArtifacts:")
    print(f"  {model_path}")
    print(f"  {metrics_path}")

    print("\n" + "=" * 70)
    print("GRAPH-ENHANCED XGBOOST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
