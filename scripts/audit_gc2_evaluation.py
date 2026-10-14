
"""
GC2 Stage 10A: Evaluation dataset and feature audit.

Checks:
- Known and unknown test distributions
- XGBoost feature compatibility
- KB feature compatibility
- Potential leakage-related fields
- Missing/nonfinite model inputs
- Dataset row overlap
- Availability of chronological timestamps

This script does not run or modify trained models.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = (
    ROOT / "data" / "processed"
    / "Data_V3_SSH_Unknown_final"
)

MODEL_PATH = (
    ROOT / "saved-models" / "xgboost_final.joblib"
)

KB_PATH = (
    ROOT / "saved-models"
    / "knn_kb"
    / "initial_kb.joblib"
)

OUTPUT_DIR = (
    ROOT / "outputs"
    / "results"
    / "gc2_evaluation"
)

SPLITS = {
    "known_train": DATA_DIR / "known_train.parquet",
    "known_validation": DATA_DIR / "known_validation.parquet",
    "known_test": DATA_DIR / "known_test.parquet",
    "unknown_test": DATA_DIR / "unknown_test.parquet",
}

SUSPICIOUS_FEATURES = [
    "id",
    "Attempted Category",
    "is_attempted",
]


def load_features():
    model = joblib.load(MODEL_PATH)

    if hasattr(model, "feature_names"):
        xgb_features = list(model.feature_names)
    else:
        inner = (
            model if hasattr(model, "get_booster")
            else model.model
        )
        xgb_features = list(
            inner.get_booster().feature_names
        )

    kb = joblib.load(KB_PATH)
    kb_features = list(kb["columns"])

    return xgb_features, kb_features


def inspect_split(name, path, xgb_features, kb_features):
    print(f"\n========== {name.upper()} ==========")

    if not path.is_file():
        raise FileNotFoundError(path)

    data = pd.read_parquet(path)

    result = {
        "split": name,
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
    }

    print("Rows:", len(data))
    print("Columns:", len(data.columns))

    if "Label" not in data.columns:
        raise ValueError(
            f"{name} has no Label column"
        )

    counts = data["Label"].astype(str).value_counts()

    print("\nClass distribution:")
    print(counts.to_string())

    result["class_distribution"] = {
        str(k): int(v)
        for k, v in counts.items()
    }

    missing_xgb = [
        f for f in xgb_features
        if f not in data.columns
    ]

    missing_kb = [
        f for f in kb_features
        if f not in data.columns
    ]

    result["missing_xgb_features"] = missing_xgb
    result["missing_kb_features"] = missing_kb

    print("\nMissing XGBoost features:", missing_xgb)
    print("Missing KB features:", missing_kb)

    if "Timestamp" in data.columns:
        times = pd.to_datetime(
            data["Timestamp"],
            errors="coerce"
        )

        result["valid_timestamps"] = int(
            times.notna().sum()
        )

        result["invalid_timestamps"] = int(
            times.isna().sum()
        )

        print(
            "Invalid timestamps:",
            result["invalid_timestamps"]
        )
    else:
        result["invalid_timestamps"] = len(data)
        print("WARNING: Timestamp column missing")

    if not missing_xgb:
        values = data[xgb_features].apply(
            pd.to_numeric,
            errors="coerce"
        ).to_numpy(dtype=np.float64)

        nonfinite = ~np.isfinite(values)

        result["nonfinite_xgb_cells"] = int(
            nonfinite.sum()
        )

        result["rows_with_nonfinite_xgb"] = int(
            nonfinite.any(axis=1).sum()
        )

        print(
            "Rows with nonfinite XGBoost inputs:",
            result["rows_with_nonfinite_xgb"]
        )

    if not missing_kb:
        values = data[kb_features].apply(
            pd.to_numeric,
            errors="coerce"
        ).to_numpy(dtype=np.float64)

        result["rows_with_nonfinite_kb"] = int(
            (~np.isfinite(values)).any(axis=1).sum()
        )

        print(
            "Rows with nonfinite KB inputs:",
            result["rows_with_nonfinite_kb"]
        )

    result["suspicious_model_features"] = [
        f for f in SUSPICIOUS_FEATURES
        if f in xgb_features
    ]

    print(
        "Potential leakage features:",
        result["suspicious_model_features"]
    )

    return data, result


def overlap_fingerprints(data, features):
    """
    Hash selected model-feature rows.

    This detects exact feature-row duplicates,
    not all forms of train/test leakage.
    """

    return pd.util.hash_pandas_object(
        data[features],
        index=False
    ).to_numpy(dtype=np.uint64)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading model feature definitions...")

    xgb_features, kb_features = load_features()

    print("XGBoost features:", len(xgb_features))
    print("KB features:", len(kb_features))

    datasets = {}
    audit = {}

    for name, path in SPLITS.items():
        data, result = inspect_split(
            name,
            path,
            xgb_features,
            kb_features
        )

        datasets[name] = data
        audit[name] = result

    # ----------------------------------------------
    # Exact feature-row overlap checks.
    # ----------------------------------------------

    print("\n========== DATASET OVERLAP ==========")

    train = datasets["known_train"]
    validation = datasets["known_validation"]
    test = datasets["known_test"]

    for name, other in (
        ("known_validation", validation),
        ("known_test", test),
    ):
        train_hashes = overlap_fingerprints(
            train,
            xgb_features
        )

        other_hashes = overlap_fingerprints(
            other,
            xgb_features
        )

        overlap = int(
            np.isin(
                other_hashes,
                train_hashes
            ).sum()
        )

        print(
            f"Known train vs {name}: "
            f"{overlap} matching feature rows"
        )

        audit[f"overlap_train_{name}"] = overlap

    # ----------------------------------------------
    # Save audit.
    # ----------------------------------------------

    output_path = (
        OUTPUT_DIR / "dataset_audit.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            audit,
            file,
            indent=2
        )

    print("\nAudit saved to:", output_path)

    print("\nGC2 DATASET AUDIT COMPLETE")


if __name__ == "__main__":
    main()
