
"""
Build a labelled historical attack reference collection.

Uses the existing Data V3 known-training split.
Does not retrain XGBoost, KNN or Autoencoder.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

MODEL_PATH = ROOT / "saved-models/xgboost_final.joblib"

TRAIN_PATH = (
    ROOT / "data/processed"
    / "Data_V3_SSH_Unknown_final"
    / "known_train.parquet"
)

OUTPUT_DIR = (
    ROOT / "data/processed/historical_profiles"
)

MAX_PER_CLASS = 100
RANDOM_STATE = 42

# Exclude fields that may encode identifiers or
# label-derived information from historical similarity.
EXCLUDED_FEATURES = {
    "id",
    "Attempted Category",
    "is_attempted",
}


def main():

    print("Loading trained XGBoost model...")

    saved = joblib.load(MODEL_PATH)

    model = (
        saved if hasattr(saved, "get_booster")
        else saved.model
    )

    feature_names = (
        list(saved.feature_names)
        if getattr(saved, "feature_names", None) is not None
        else list(model.get_booster().feature_names)
    )

    comparison_features = [
        feature
        for feature in feature_names
        if feature not in EXCLUDED_FEATURES
    ]

    print("Classifier features:", len(feature_names))
    print("Comparison features:", len(comparison_features))

    print("Loading known-training dataset...")

    train = pd.read_parquet(TRAIN_PATH)

    required = comparison_features + ["Label"]

    missing = [
        column
        for column in required
        if column not in train.columns
    ]

    if missing:
        raise ValueError(
            f"Training data missing columns: {missing}"
        )

    attacks = train.loc[
        train["Label"].astype(str).str.upper() != "BENIGN"
    ].copy()

    print("Available attack records:", len(attacks))

    sampled = []

    for label, group in attacks.groupby("Label"):

        n = min(MAX_PER_CLASS, len(group))

        selection = group.sample(
            n=n,
            random_state=RANDOM_STATE
        )

        sampled.append(selection)

        print(f"{label}: {n} reference records")

    if not sampled:
        raise ValueError(
            "No labelled attack records available"
        )

    profiles = pd.concat(
        sampled,
        ignore_index=True
    )

    profiles = profiles[
        comparison_features + ["Label"]
    ].copy()

    for feature in comparison_features:
        profiles[feature] = pd.to_numeric(
            profiles[feature],
            errors="raise"
        )

    values = profiles[
        comparison_features
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            "Historical profiles contain nonfinite features"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    profiles.to_parquet(
        OUTPUT_DIR / "attack_profiles.parquet",
        index=False
    )

    joblib.dump(
        comparison_features,
        OUTPUT_DIR / "comparison_features.joblib"
    )

    print("\nTotal reference records:", len(profiles))
    print("Saved historical attack profiles.")
    print("HISTORICAL PROFILE BUILD: PASSED")


if __name__ == "__main__":
    main()
