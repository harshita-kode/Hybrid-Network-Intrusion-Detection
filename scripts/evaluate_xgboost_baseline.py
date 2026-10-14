
"""
GC2 Stage 10B: XGBoost baseline evaluation.

Evaluates the saved XGBoost classifier on the complete
Data V3 known-test split.

No retraining, KB updates or model modifications.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

MODEL_PATH = (
    ROOT / "saved-models" / "xgboost_final.joblib"
)

TEST_PATH = (
    ROOT / "data/processed"
    / "Data_V3_SSH_Unknown_final"
    / "known_test.parquet"
)

OUTPUT_DIR = (
    ROOT / "outputs/results/gc2_evaluation/xgboost"
)

CONFIDENCE_THRESHOLD = 0.95

BATCH_SIZE = 10000


def load_classifier():

    saved = joblib.load(MODEL_PATH)

    model = (
        saved if hasattr(saved, "get_booster")
        else saved.model
    )

    if getattr(saved, "feature_names", None) is not None:
        features = list(saved.feature_names)
    else:
        features = list(
            model.get_booster().feature_names
        )

    if len(features) != 87:
        raise ValueError(
            f"Expected 87 features, got {len(features)}"
        )

    if model.n_classes_ != 13:
        raise ValueError(
            "Expected 13 classifier classes"
        )

    return saved, model, features


def resolve_class_names(saved, model, train_path):
    """
    Prefer the saved classifier's own class mapping.

    For a numeric 0..12 encoding, reconstruct the mapping
    from known_train labels and verify its compatibility.

    IMPORTANT:
    This fallback assumes the training code used
    sklearn LabelEncoder's sorted class order.
    """

    classes = np.asarray(model.classes_)

    if len(classes) != 13:
        raise ValueError("Unexpected class mapping")

    if all(isinstance(x, str) for x in classes):
        return list(classes)

    train_labels = pd.read_parquet(
        train_path,
        columns=["Label"]
    )["Label"].astype(str)

    names = sorted(train_labels.unique().tolist())

    if len(names) != 13:
        raise ValueError(
            "Training labels do not contain 13 classes"
        )

    if not np.array_equal(
        classes.astype(int),
        np.arange(13)
    ):
        raise ValueError(
            "Numeric class IDs are not 0..12"
        )

    print(
        "WARNING: Using sorted known_train labels "
        "as the numeric class mapping. Confirm this "
        "matches the original training encoder."
    )

    return names


def predict_batches(model, X, batch_size):

    probabilities = []

    start = time.perf_counter()

    for beginning in range(0, len(X), batch_size):

        ending = min(
            beginning + batch_size,
            len(X)
        )

        batch = X.iloc[beginning:ending]

        proba = model.predict_proba(batch)

        probabilities.append(
            np.asarray(proba, dtype=np.float64)
        )

        print(
            f"Predicted {ending:,}/{len(X):,} flows"
        )

    elapsed = time.perf_counter() - start

    probabilities = np.vstack(probabilities)

    return probabilities, elapsed


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading XGBoost classifier...")

    saved, model, features = load_classifier()

    train_path = (
        ROOT / "data/processed"
        / "Data_V3_SSH_Unknown_final"
        / "known_train.parquet"
    )

    class_names = resolve_class_names(
        saved,
        model,
        train_path
    )

    print("Loading known-test dataset...")

    test = pd.read_parquet(TEST_PATH)

    missing = [
        feature for feature in features
        if feature not in test.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    X = test[features].copy()

    y_true = test["Label"].astype(str).to_numpy()

    unknown_labels = set(y_true) - set(class_names)

    if unknown_labels:
        raise ValueError(
            f"Unexpected known-test labels: {unknown_labels}"
        )

    print("Test samples:", len(X))
    print("Model features:", len(features))
    print("Classes:", len(class_names))

    print("\nRunning batch predictions...")

    probabilities, elapsed = predict_batches(
        model,
        X,
        BATCH_SIZE
    )

    if probabilities.shape != (len(X), len(class_names)):
        raise ValueError(
            "Prediction probability shape mismatch"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Nonfinite prediction probabilities"
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-4
    ):
        raise ValueError(
            "Prediction probabilities do not sum to one"
        )

    predicted_indices = np.argmax(
        probabilities,
        axis=1
    )

    y_pred = np.asarray(class_names)[
        predicted_indices
    ]

    confidence = probabilities.max(axis=1)

    high_confidence = (
        confidence >= CONFIDENCE_THRESHOLD
    )

    low_confidence = ~high_confidence

    correct = y_true == y_pred

    # ----------------------------------------------
    # Overall classification metrics.
    # ----------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=class_names,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=class_names,
            average="weighted",
            zero_division=0,
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).T

    report_df.to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=class_names
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names
    )

    matrix_df.to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    # ----------------------------------------------
    # Confidence routing metrics.
    # ----------------------------------------------

    high_count = int(high_confidence.sum())
    low_count = int(low_confidence.sum())

    high_accuracy = (
        float(correct[high_confidence].mean())
        if high_count
        else None
    )

    low_error_rate = (
        float((~correct[low_confidence]).mean())
        if low_count
        else None
    )

    routing = {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "high_confidence_count": high_count,
        "low_confidence_count": low_count,
        "high_confidence_coverage": (
            high_count / len(X)
        ),
        "low_confidence_rate": (
            low_count / len(X)
        ),
        "high_confidence_accuracy": high_accuracy,
        "low_confidence_error_rate": low_error_rate,
    }

    # ----------------------------------------------
    # Per-flow prediction output.
    # ----------------------------------------------

    results = pd.DataFrame({
        "test_row": np.arange(len(test)),
        "actual_class": y_true,
        "predicted_class": y_pred,
        "confidence": confidence,
        "correct": correct,
        "confidence_route": np.where(
            high_confidence,
            "HIGH_CONFIDENCE",
            "LOW_CONFIDENCE"
        ),
    })

    results.to_csv(
        OUTPUT_DIR / "known_predictions.csv",
        index=False
    )

    # ----------------------------------------------
    # Save metrics.
    # ----------------------------------------------

    metrics = {
        "dataset": "Data_V3_known_test",
        "samples": int(len(test)),
        "features": len(features),
        "classes": len(class_names),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "prediction_seconds": float(elapsed),
        "flows_per_second": float(
            len(test) / elapsed
        ) if elapsed > 0 else None,
        "routing": routing,
        "evaluation_limitations": [
            "XGBoost-only evaluation, not full GC2 accuracy.",
            "Potential leakage features require review.",
            "Class mapping must match original training encoder.",
            "Batch prediction timing excludes model loading and SHAP.",
        ],
    }

    with (
        OUTPUT_DIR / "baseline_metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            indent=2,
            allow_nan=False
        )

    # ----------------------------------------------
    # Terminal report.
    # ----------------------------------------------

    print("\n========== XGBOOST BASELINE ==========")

    print("Samples:", len(test))
    print(f"Accuracy: {accuracy:.4%}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\n========== CONFIDENCE ROUTING ==========")

    print("High-confidence flows:", high_count)
    print("Low-confidence flows:", low_count)

    print(
        "High-confidence coverage:",
        f"{high_count / len(test):.4%}"
    )

    print(
        "High-confidence accuracy:",
        f"{high_accuracy:.4%}"
        if high_accuracy is not None
        else "N/A"
    )

    print(
        "Low-confidence error rate:",
        f"{low_error_rate:.4%}"
        if low_error_rate is not None
        else "N/A"
    )

    print("\n========== EXECUTION ==========")

    print(f"Prediction time: {elapsed:.3f} seconds")
    print(
        f"Throughput: {len(test) / elapsed:.1f} flows/sec"
    )

    print("\n========== PER-CLASS RESULTS ==========")

    print(
        report_df.loc[
            class_names,
            ["precision", "recall", "f1-score", "support"]
        ].to_string()
    )

    print("\nResults saved to:", OUTPUT_DIR)
    print("\nXGBOOST BASELINE EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
