
"""Find genuine low-confidence KB matches and misses."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from notebooks.kb.kb_flow import search
from models.confidence import DEFAULT_CONFIDENCE_THRESHOLD


MODEL_PATH = ROOT / "saved-models/xgboost_final.joblib"
KB_PATH = ROOT / "saved-models/knn_kb/initial_kb.joblib"
REFERENCE_PATH = (
    ROOT / "data/processed/knn_kb/knowledge_base.csv"
)
TEST_PATH = (
    ROOT / "data/processed/Data_V3_SSH_Unknown_final"
    / "known_test.parquet"
)
OUTPUT_DIR = ROOT / "data/incoming"


def main():
    print("Loading models and data...")

    saved = joblib.load(MODEL_PATH)
    model = (
        saved if hasattr(saved, "get_booster")
        else saved.model
    )

    features = (
        list(saved.feature_names)
        if getattr(saved, "feature_names", None) is not None
        else list(model.get_booster().feature_names)
    )

    kb = joblib.load(KB_PATH)
    reference = pd.read_csv(
        REFERENCE_PATH, low_memory=False
    )
    test = pd.read_parquet(TEST_PATH)

    if len(reference) != kb["count"]:
        raise ValueError("KB index/reference count mismatch")

    low_indices = []

    for start in range(0, len(test), 10000):
        batch = test.iloc[start:start + 10000]
        probabilities = model.predict_proba(batch[features])
        confidence = probabilities.max(axis=1)

        indices = np.flatnonzero(
            confidence < DEFAULT_CONFIDENCE_THRESHOLD
        )

        low_indices.extend(
            (start + indices).tolist()
        )

    print("Low-confidence records:", len(low_indices))

    if not low_indices:
        return

    # Search the KB only for genuinely low-confidence flows.
    results = search(
        test.iloc[low_indices],
        kb,
        reference,
        k=5,
        trace=False
    )

    if len(results) != len(low_indices):
        raise ValueError("KB result count mismatch")

    matches = []
    misses = []

    for position, result in zip(low_indices, results):
        record = {
            "known_test_row": int(position),
            "kb_matched": bool(result["matched"]),
            "nearest_distance": float(
                result["nearest_distance"]
            ),
        }

        if result["matched"]:
            matches.append(record)
        else:
            misses.append(record)

    print("KB matches:", len(matches))
    print("KB misses:", len(misses))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(matches, columns=[
        "known_test_row",
        "kb_matched",
        "nearest_distance"
    ]).to_csv(
        OUTPUT_DIR / "kb_matches_audit.csv",
        index=False
    )

    if matches:
        positions = [
            item["known_test_row"]
            for item in matches[:5]
        ]

        test.iloc[positions].to_csv(
            OUTPUT_DIR / "kb_matched_traffic.csv",
            index=False
        )

        print("Matched test rows:", positions)

    if misses:
        positions = [
            item["known_test_row"]
            for item in misses[:5]
        ]

        test.iloc[positions].to_csv(
            OUTPUT_DIR / "kb_unmatched_traffic.csv",
            index=False
        )

    print("KB search verification completed.")


if __name__ == "__main__":
    main()
