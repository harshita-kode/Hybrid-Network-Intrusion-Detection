
"""Find real low-confidence XGBoost test flows."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.confidence import DEFAULT_CONFIDENCE_THRESHOLD


MODEL_PATH = ROOT / "saved-models/xgboost_final.joblib"

TEST_PATH = (
    ROOT / "data/processed"
    / "Data_V3_SSH_Unknown_final"
    / "known_test.parquet"
)

OUTPUT_PATH = (
    ROOT / "data/incoming"
    / "low_confidence_traffic.csv"
)


def main():

    print("Loading XGBoost model...")

    saved = joblib.load(MODEL_PATH)

    model = (
        saved if hasattr(saved, "get_booster")
        else saved.model
    )

    feature_names = list(
        saved.feature_names
        if hasattr(saved, "feature_names")
        and saved.feature_names is not None
        else model.get_booster().feature_names
    )

    print("Loading test data...")

    test = pd.read_parquet(TEST_PATH)

    missing = [
        name for name in feature_names
        if name not in test.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    X = test[feature_names]

    print("Calculating prediction confidence...")

    # Predict in batches to limit temporary memory usage.
    batch_size = 10000
    selected_indices = []
    selected_confidences = []

    for start in range(0, len(X), batch_size):

        end = min(start + batch_size, len(X))

        probabilities = model.predict_proba(
            X.iloc[start:end]
        )

        confidence = np.max(
            probabilities,
            axis=1
        )

        low_positions = np.flatnonzero(
            confidence < DEFAULT_CONFIDENCE_THRESHOLD
        )

        for position in low_positions:
            selected_indices.append(
                start + int(position)
            )

            selected_confidences.append(
                float(confidence[position])
            )

    print(
        "Low-confidence records:",
        len(selected_indices)
    )

    if not selected_indices:
        print(
            "No low-confidence samples found "
            "at the configured threshold."
        )
        return

    # Select the five least-confident records.
    ranking = np.argsort(
        selected_confidences
    )[:5]

    chosen = [
        selected_indices[i]
        for i in ranking
    ]

    selected = test.iloc[chosen].copy()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    selected.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSelected test positions:", chosen)

    print(
        "Selected confidence values:",
        [selected_confidences[i] for i in ranking]
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
