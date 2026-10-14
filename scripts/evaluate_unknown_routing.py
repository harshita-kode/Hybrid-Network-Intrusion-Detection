
"""
GC2 Stage 10C: Unknown attack confidence-routing evaluation.

Evaluates all held-out unknown-test flows using
the existing frozen XGBoost classifier.

Measures:
- High-confidence unknown flows
- Low-confidence unknown flows
- High-confidence benign predictions
- High-confidence known-attack predictions
- Routing by unknown attack family
- Confidence distributions
- Exploratory threshold sensitivity

Does not modify trained models or KB artifacts.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

MODEL_PATH = (
    ROOT / "saved-models"
    / "xgboost_final.joblib"
)

UNKNOWN_PATH = (
    ROOT / "data"
    / "processed"
    / "Data_V3_SSH_Unknown_final"
    / "unknown_test.parquet"
)

OUTPUT_DIR = (
    ROOT / "outputs"
    / "results"
    / "gc2_evaluation"
    / "unknown_routing"
)

CONFIDENCE_THRESHOLD = 0.95
BATCH_SIZE = 5000

# Exploratory thresholds only.
# These do not change the deployed GC2 threshold.
THRESHOLDS = [0.80, 0.90, 0.95, 0.99]


def load_classifier():

    saved = joblib.load(MODEL_PATH)

    model = (
        saved
        if hasattr(saved, "get_booster")
        else saved.model
    )

    features = getattr(
        saved,
        "feature_names",
        None
    )

    if features is None:
        features = model.get_booster().feature_names

    if features is None:
        raise ValueError(
            "Saved model has no feature names"
        )

    features = list(features)

    if len(features) != 87:
        raise ValueError(
            f"Expected 87 features, got {len(features)}"
        )

    if model.n_classes_ != 13:
        raise ValueError(
            "Expected 13 classifier classes"
        )

    return saved, model, features


def resolve_class_names(model):

    # This mapping matches the class mapping printed
    # by the verified GC2 SHAP integration.
    #
    # It should still be checked against the original
    # training encoder before final publication.

    names = [
        "BENIGN",
        "Botnet",
        "DDoS",
        "DoS GoldenEye",
        "DoS Hulk",
        "DoS Slowloris",
        "Heartbleed",
        "Infiltration",
        "Infiltration - Portscan",
        "Portscan",
        "Web Attack - Brute Force",
        "Web Attack - SQL Injection",
        "Web Attack - XSS",
    ]

    classes = np.asarray(model.classes_)

    if not np.array_equal(
        classes,
        np.arange(len(names))
    ):
        raise ValueError(
            "Unexpected model class IDs. "
            "Verify the original label encoder."
        )

    return names


def predict_batches(model, X):

    predictions = []

    start = time.perf_counter()

    for beginning in range(
        0,
        len(X),
        BATCH_SIZE
    ):

        ending = min(
            beginning + BATCH_SIZE,
            len(X)
        )

        batch = X.iloc[
            beginning:ending
        ]

        probabilities = model.predict_proba(
            batch
        )

        predictions.append(
            np.asarray(
                probabilities,
                dtype=np.float64
            )
        )

        print(
            f"Predicted {ending:,}/{len(X):,} flows"
        )

    elapsed = time.perf_counter() - start

    return np.vstack(predictions), elapsed


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading trained XGBoost model...")

    saved, model, features = load_classifier()

    class_names = resolve_class_names(model)

    print("Loading held-out unknown-test dataset...")

    data = pd.read_parquet(UNKNOWN_PATH)

    required = features + ["Label"]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    actual = data["Label"].astype(str).to_numpy()

    overlapping_labels = (
        set(actual) & set(class_names)
    )

    if overlapping_labels:
        raise ValueError(
            "Unknown-test contains known labels: "
            f"{overlapping_labels}"
        )

    X = data[features].copy()

    print("Unknown samples:", len(data))
    print("Model features:", len(features))

    print("\nRunning predictions...")

    probabilities, elapsed = predict_batches(
        model,
        X
    )

    if probabilities.shape != (
        len(data),
        len(class_names)
    ):
        raise ValueError(
            "Unexpected probability shape"
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
            "Probabilities do not sum to one"
        )

    predicted_indices = np.argmax(
        probabilities,
        axis=1
    )

    predicted = np.asarray(
        class_names
    )[predicted_indices]

    confidence = probabilities.max(
        axis=1
    )

    low_confidence = (
        confidence < CONFIDENCE_THRESHOLD
    )

    high_confidence = ~low_confidence

    # --------------------------------------------
    # Save individual predictions.
    # --------------------------------------------

    results = pd.DataFrame({
        "test_row": np.arange(len(data)),
        "actual_unknown_class": actual,
        "predicted_known_class": predicted,
        "confidence": confidence,
        "confidence_route": np.where(
            low_confidence,
            "LOW_CONFIDENCE",
            "HIGH_CONFIDENCE"
        ),
        "routed_to_kb": low_confidence,
        "accepted_as_benign": (
            high_confidence
            & (predicted == "BENIGN")
        ),
        "accepted_as_known_attack": (
            high_confidence
            & (predicted != "BENIGN")
        ),
    })

    results.to_csv(
        OUTPUT_DIR / "unknown_predictions.csv",
        index=False
    )

    # --------------------------------------------
    # Overall routing counts.
    # --------------------------------------------

    total = len(results)

    low_count = int(
        low_confidence.sum()
    )

    high_count = int(
        high_confidence.sum()
    )

    high_benign = int(
        results["accepted_as_benign"].sum()
    )

    high_attack = int(
        results["accepted_as_known_attack"].sum()
    )

    assert (
        low_count + high_benign + high_attack
        == total
    )

    # --------------------------------------------
    # Per-unknown-class routing.
    # --------------------------------------------

    per_class = (
        results.groupby(
            "actual_unknown_class"
        )
        .agg(
            samples=("test_row", "size"),
            low_confidence_count=(
                "routed_to_kb",
                "sum"
            ),
            accepted_benign_count=(
                "accepted_as_benign",
                "sum"
            ),
            accepted_known_attack_count=(
                "accepted_as_known_attack",
                "sum"
            ),
            mean_confidence=(
                "confidence",
                "mean"
            ),
            median_confidence=(
                "confidence",
                "median"
            ),
        )
        .reset_index()
    )

    per_class["low_confidence_rate"] = (
        per_class["low_confidence_count"]
        / per_class["samples"]
    )

    per_class["accepted_benign_rate"] = (
        per_class["accepted_benign_count"]
        / per_class["samples"]
    )

    per_class.to_csv(
        OUTPUT_DIR / "per_unknown_class.csv",
        index=False
    )

    # --------------------------------------------
    # Threshold sensitivity.
    # --------------------------------------------

    sensitivity = []

    for threshold in THRESHOLDS:

        routed = confidence < threshold

        sensitivity.append({
            "threshold": threshold,
            "routed_count": int(
                routed.sum()
            ),
            "routed_rate": float(
                routed.mean()
            ),
            "high_confidence_benign_count": int(
                (
                    ~routed
                    & (predicted == "BENIGN")
                ).sum()
            ),
        })

    sensitivity_df = pd.DataFrame(
        sensitivity
    )

    sensitivity_df.to_csv(
        OUTPUT_DIR / "threshold_sensitivity.csv",
        index=False
    )

    # --------------------------------------------
    # Summary metrics.
    # --------------------------------------------

    metrics = {
        "dataset": "Data_V3_unknown_test",
        "samples": total,
        "unknown_classes": sorted(
            set(actual)
        ),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "low_confidence_count": low_count,
        "low_confidence_rate": (
            low_count / total
        ),
        "high_confidence_count": high_count,
        "high_confidence_rate": (
            high_count / total
        ),
        "high_confidence_benign_count": high_benign,
        "high_confidence_benign_rate": (
            high_benign / total
        ),
        "high_confidence_known_attack_count": high_attack,
        "high_confidence_known_attack_rate": (
            high_attack / total
        ),
        "prediction_seconds": float(elapsed),
        "interpretation": (
            "Low-confidence routing is an investigation "
            "rate, not unknown-attack detection recall."
        ),
        "limitations": [
            "XGBoost routing only; KB and AE not evaluated.",
            "Class mapping must match original training encoder.",
            "Potential identifier or label leakage unresolved.",
            "Unknown-test independence from KB and AE training not yet verified.",
        ],
    }

    with (
        OUTPUT_DIR / "unknown_routing_metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            indent=2,
            allow_nan=False
        )

    # --------------------------------------------
    # Terminal output.
    # --------------------------------------------

    print("\n========== UNKNOWN ROUTING ==========")

    print("Total unknown flows:", total)

    print(
        "Low-confidence / KB route:",
        low_count,
        f"({low_count / total:.2%})"
    )

    print(
        "High-confidence known attack:",
        high_attack,
        f"({high_attack / total:.2%})"
    )

    print(
        "High-confidence BENIGN:",
        high_benign,
        f"({high_benign / total:.2%})"
    )

    print("\n========== PER-CLASS ROUTING ==========")

    print(
        per_class.to_string(
            index=False
        )
    )

    print("\n========== THRESHOLD SENSITIVITY ==========")

    print(
        sensitivity_df.to_string(
            index=False
        )
    )

    print("\n========== EXECUTION ==========")

    print(
        f"Prediction time: {elapsed:.3f} seconds"
    )

    print("\nResults saved to:", OUTPUT_DIR)

    print(
        "\nUNKNOWN ROUTING EVALUATION COMPLETE"
    )


if __name__ == "__main__":
    main()
