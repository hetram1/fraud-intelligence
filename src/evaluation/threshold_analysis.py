from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy import sparse
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "features"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"

MODEL_FILE = MODEL_DIR / "xgboost_baseline.json"
BASELINE_METRICS_FILE = (
    EVALUATION_DIR / "xgboost_baseline_validation.json"
)

THRESHOLDS = np.round(
    np.arange(0.05, 1.00, 0.05),
    2,
)


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def main() -> None:
    print("=" * 70)
    print("VALIDATION THRESHOLD ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load validation data
    # ---------------------------------------------------------------
    X_path = FEATURE_DIR / "X_validation.npz"
    y_path = FEATURE_DIR / "y_validation.npy"

    if not X_path.exists():
        fail(f"Missing validation matrix: {X_path}")

    if not y_path.exists():
        fail(f"Missing validation target: {y_path}")

    if not MODEL_FILE.exists():
        fail(f"Missing model: {MODEL_FILE}")

    if not BASELINE_METRICS_FILE.exists():
        fail(
            f"Missing baseline metrics: "
            f"{BASELINE_METRICS_FILE}"
        )

    X_validation = sparse.load_npz(X_path).tocsr()
    y_validation = np.load(y_path).astype(bool)

    with BASELINE_METRICS_FILE.open() as f:
        baseline = json.load(f)

    best_iteration = baseline.get("best_iteration")

    print(
        f"Validation matrix : "
        f"{X_validation.shape[0]:,} x "
        f"{X_validation.shape[1]:,}"
    )

    print(f"Validation fraud  : {int(y_validation.sum()):,}")
    print(f"Best iteration    : {best_iteration}")

    # ---------------------------------------------------------------
    # Load the same XGBoost model that produced the baseline.
    # ---------------------------------------------------------------
    model = xgb.Booster()
    model.load_model(MODEL_FILE)

    dvalidation = xgb.DMatrix(X_validation)

    if best_iteration is not None:
        probabilities = model.predict(
            dvalidation,
            iteration_range=(0, int(best_iteration) + 1),
        )
    else:
        probabilities = model.predict(dvalidation)

    # ---------------------------------------------------------------
    # Verify ranking metrics independently.
    # ---------------------------------------------------------------
    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    print("\nRanking metrics:")
    print(f"  PR-AUC  : {pr_auc:.6f}")
    print(f"  ROC-AUC : {roc_auc:.6f}")

    # ---------------------------------------------------------------
    # Evaluate multiple operating thresholds.
    # ---------------------------------------------------------------
    results = []

    print("\nThreshold table:")
    print(
        f"{'Threshold':>10} "
        f"{'Precision':>11} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'Flagged %':>11}"
    )

    for threshold in THRESHOLDS:
        predictions = probabilities >= threshold

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        flagged_percentage = (
            predictions.mean() * 100
        )

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            predictions,
            labels=[False, True],
        ).ravel()

        row = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "flagged_percentage": float(
                flagged_percentage
            ),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

        results.append(row)

        print(
            f"{threshold:>10.2f} "
            f"{precision:>11.4f} "
            f"{recall:>10.4f} "
            f"{f1:>10.4f} "
            f"{flagged_percentage:>10.2f}%"
        )

    # ---------------------------------------------------------------
    # Select the F1-maximizing validation threshold as an analysis
    # result. This is NOT a final business policy.
    # ---------------------------------------------------------------
    f1_best = max(
        results,
        key=lambda row: (
            row["f1"],
            row["recall"],
            -row["threshold"],
        ),
    )

    print("\nValidation F1-maximizing threshold:")
    print(f"  Threshold : {f1_best['threshold']:.2f}")
    print(f"  Precision : {f1_best['precision']:.6f}")
    print(f"  Recall    : {f1_best['recall']:.6f}")
    print(f"  F1        : {f1_best['f1']:.6f}")
    print(
        f"  Flagged % : "
        f"{f1_best['flagged_percentage']:.2f}%"
    )

    # ---------------------------------------------------------------
    # Save analysis.
    # ---------------------------------------------------------------
    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "model": "xgboost_baseline",
        "evaluation_split": "validation",
        "selection_metric": "f1",
        "note": (
            "The threshold selected here is an analytical "
            "validation result, not a final production policy. "
            "Production threshold selection should incorporate "
            "investigation capacity, false-positive cost, "
            "missed-fraud cost, and calibration."
        ),
        "best_iteration": (
            int(best_iteration)
            if best_iteration is not None
            else None
        ),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "thresholds": results,
        "f1_max_threshold": f1_best,
    }

    output_path = (
        EVALUATION_DIR
        / "threshold_analysis_validation.json"
    )

    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print("\nSaved:")
    print(f"  {output_path}")

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
