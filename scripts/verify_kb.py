
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notebooks.kb.kb_flow import read_partition
from src.analysis.kb_explanation import explain_kb_match


DATA = ROOT / "data" / "processed"
SPLITS = (
    DATA / "Data_V3_SSH_Unknown_final"
    / "unknown_kb_splits"
)
KB = DATA / "knn_kb"


def to_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).strip().lower() in ("true", "1"):
        return True
    if str(value).strip().lower() in ("false", "0"):
        return False
    raise ValueError(f"Invalid match value: {value}")


def main():
    artifact = joblib.load(
        ROOT / "saved-models/knn_kb/initial_kb.joblib"
    )
    reference = pd.read_csv(
        KB / "knowledge_base.csv",
        low_memory=False
    )
    demo = read_partition(
        SPLITS / "unknown_demo_kb_updates.parquet"
    )
    audit = pd.read_csv(
        KB / "demo_similarity_audit"
        / "demo_similarity_audit.csv"
    )

    required = {
        "demo_row", "matched",
        "nearest_distance", "nearest_pattern_id"
    }
    if not required.issubset(audit.columns):
        raise ValueError(
            f"Missing audit columns: {required - set(audit.columns)}"
        )

    if audit["demo_row"].duplicated().any():
        raise ValueError("Duplicate demo_row values")

    # Test up to 20 matches and 20 misses.
    matched = audit[
        audit["matched"].map(to_bool)
    ].head(20)

    missed = audit[
        ~audit["matched"].map(to_bool)
    ].head(20)

    selected = pd.concat([matched, missed])

    if matched.empty or missed.empty:
        raise ValueError(
            "Both KB-match and KB-miss samples are required"
        )

    passed = 0

    for _, row in selected.iterrows():
        index = int(row["demo_row"])

        result = explain_kb_match(
            flow=demo.iloc[[index]],
            artifact=artifact,
            reference=reference,
            top_n=5
        )

        assert result["kb_matched"] == to_bool(
            row["matched"]
        )

        assert np.isclose(
            result["nearest_distance"],
            float(row["nearest_distance"]),
            rtol=1e-5,
            atol=1e-7
        )

        assert str(result["nearest_pattern_id"]) == str(
            row["nearest_pattern_id"]
        )

        distance = result["nearest_distance"]
        cutoff = result["match_cutoff"]

        assert result["kb_matched"] == (
            distance <= cutoff
        )

        # The top-five contributions must never
        # exceed the full squared distance.
        top_sum = sum(
            item["squared_difference"]
            for item in result["top_feature_differences"]
        )

        assert top_sum <= distance ** 2 + 1e-6

        passed += 1

    print("Records tested:", passed)
    print("Matched tested:", len(matched))
    print("Missed tested:", len(missed))
    print("KNN AUDIT VERIFICATION: PASSED")


if __name__ == "__main__":
    main()
