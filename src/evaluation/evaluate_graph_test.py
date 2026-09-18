from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.sparse import load_npz


ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    ROOT / "data" / "processed" / "graph_enhanced_features"
)

MODEL_PATH = (
    ROOT
    / "outputs"
    / "models"
    / "xgboost_graph_enhanced.json"
)

OUTPUT_PATH = (
    ROOT
    / "outputs"
    / "evaluation"
    / "xgboost_graph_enhanced_test.json"
)

FIXED_THRESHOLD = 0.50


def main() -> None:
    print("=" * 70)
    print("XGBOOST GRAPH-ENHANCED — UNTOUCHED TEST SET EVALUATION")
    print("=" * 70)

    test_x_path = FEATURE_DIR / "X_test.npz"
    test_y_path = FEATURE_DIR / "y_test.npy"

    if not test_x_path.exists():
        raise FileNotFoundError(
            f"Missing test features: {test_x_path}"
        )

    if not test_y_path.exists():
        raise FileNotFoundError(
            f"Missing test labels: {test_y_path}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}"
        )

    X_test = load_npz(test_x_path)
    y_test = np.load(test_y_path).astype(bool)

    print(f"\nTest matrix shape: {X_test.shape}")
    print(f"Test labels:       {y_test.shape}")

    model = xgb.Booster()
    model.load_model(MODEL_PATH)

    best_iteration = int(model.best_iteration)

    print(
        f"Best iteration from validation: "
        f"{best_iteration}"
    )

    dtest = xgb.DMatrix(X_test)

    probabilities = model.predict(
        dtest,
        iteration_range=(0, best_iteration + 1),
    )

    predictions = (
        probabilities >= FIXED_THRESHOLD
    ).astype(int)

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    flagged_rate = float(
        predictions.mean()
    )

    results = {
        "dataset": "test",
        "model": "xgboost_graph_enhanced",
        "test_rows": int(len(y_test)),
        "feature_count": int(X_test.shape[1]),
        "best_iteration_from_validation": best_iteration,
        "threshold": FIXED_THRESHOLD,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "flagged_rate": flagged_rate,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print("\nTEST SET RESULTS")
    print("-" * 70)
    print(f"PR-AUC:           {pr_auc:.6f}")
    print(f"ROC-AUC:          {roc_auc:.6f}")
    print(
        f"Precision @ 0.50: {precision:.6f}"
    )
    print(
        f"Recall @ 0.50:    {recall:.6f}"
    )
    print(
        f"F1 @ 0.50:        {f1:.6f}"
    )
    print(
        f"Flagged rate:     {flagged_rate:.2%}"
    )

    print("\nConfusion Matrix")
    print("-" * 70)
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")

    print(
        f"\nSaved results to: "
        f"{OUTPUT_PATH}"
    )

    print("\nGRAPH TEST EVALUATION: PASS")


if __name__ == "__main__":
    main()
