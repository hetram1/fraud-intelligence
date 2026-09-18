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

FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "features"
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
    name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    predictions = probabilities >= 0.5

    metrics = {
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
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

    print(f"\n{name} metrics:")
    print(f"  PR-AUC          : {metrics['pr_auc']:.6f}")
    print(f"  ROC-AUC         : {metrics['roc_auc']:.6f}")
    print(
        f"  Precision @ 0.5 : "
        f"{metrics['precision_at_0_5']:.6f}"
    )
    print(
        f"  Recall @ 0.5    : "
        f"{metrics['recall_at_0_5']:.6f}"
    )
    print(
        f"  F1 @ 0.5        : "
        f"{metrics['f1_at_0_5']:.6f}"
    )

    return metrics


def main() -> None:
    print("=" * 70)
    print("XGBOOST FRAUD BASELINE")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load the already-prepared sparse matrices.
    # ---------------------------------------------------------------
    X_train = load_matrix("train")
    X_validation = load_matrix("validation")

    y_train = load_target("train")
    y_validation = load_target("validation")

    print(f"Train matrix      : {X_train.shape}")
    print(f"Validation matrix : {X_validation.shape}")

    print(f"\nTrain fraud count      : {int(y_train.sum()):,}")
    print(
        f"Train non-fraud count  : "
        f"{int((~y_train).sum()):,}"
    )

    # ---------------------------------------------------------------
    # Handle class imbalance.
    #
    # scale_pos_weight = negative / positive
    # ---------------------------------------------------------------
    positive_count = int(y_train.sum())
    negative_count = int((~y_train).sum())

    if positive_count == 0:
        fail("Training set contains no positive fraud examples.")

    scale_pos_weight = negative_count / positive_count

    print(
        f"Scale positive weight : "
        f"{scale_pos_weight:.6f}"
    )

    # ---------------------------------------------------------------
    # Create XGBoost matrices.
    # ---------------------------------------------------------------
    dtrain = xgb.DMatrix(
        X_train,
        label=y_train.astype(np.float32),
    )

    dvalidation = xgb.DMatrix(
        X_validation,
        label=y_validation.astype(np.float32),
    )

    # ---------------------------------------------------------------
    # Model configuration.
    #
    # We start with a strong but controlled baseline.
    # ---------------------------------------------------------------
    params = {
        "objective": "binary:logistic",
        # PR-AUC is the primary early-stopping metric because fraud
        # detection is an imbalanced classification problem.
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

    evaluation_history: dict[str, dict[str, list[float]]] = {}

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

    if best_iteration is not None:
        print(f"Best iteration: {best_iteration}")

    # ---------------------------------------------------------------
    # Validation prediction.
    # ---------------------------------------------------------------
    validation_probabilities = model.predict(
        dvalidation,
        iteration_range=(
            0,
            model.best_iteration + 1
            if model.best_iteration is not None
            else 1000,
        ),
    )

    validation_metrics = evaluate(
        "Validation",
        y_validation,
        validation_probabilities,
    )

    # ---------------------------------------------------------------
    # Save generated artifacts locally.
    # These are intentionally ignored by Git.
    # ---------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "xgboost_baseline.json"

    model.save_model(model_path)

    metrics = {
        "model": "xgboost_baseline",
        "random_state": RANDOM_STATE,
        "best_iteration": (
            int(model.best_iteration)
            if model.best_iteration is not None
            else None
        ),
        "scale_pos_weight": float(scale_pos_weight),
        "training_rows": int(X_train.shape[0]),
        "validation_rows": int(X_validation.shape[0]),
        "feature_count": int(X_train.shape[1]),
        "validation_metrics": validation_metrics,
        "evaluation_history": evaluation_history,
    }

    metrics_path = EVALUATION_DIR / "xgboost_baseline_validation.json"

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\nArtifacts:")
    print(f"  {model_path}")
    print(f"  {metrics_path}")

    print("\n" + "=" * 70)
    print("XGBOOST BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
