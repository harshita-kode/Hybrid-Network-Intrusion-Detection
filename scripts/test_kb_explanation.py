
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notebooks.kb.kb_flow import read_partition
from src.analysis.kb_explanation import explain_kb_match


DATA_DIR = ROOT / "data" / "processed"

SPLITS_DIR = (
    DATA_DIR
    / "Data_V3_SSH_Unknown_final"
    / "unknown_kb_splits"
)

KB_DIR = DATA_DIR / "knn_kb"

ARTIFACT_PATH = (
    ROOT / "saved-models"
    / "knn_kb"
    / "initial_kb.joblib"
)

REFERENCE_PATH = KB_DIR / "knowledge_base.csv"

AUDIT_PATH = (
    KB_DIR
    / "demo_similarity_audit"
    / "demo_similarity_audit.csv"
)


def main():
    print("Loading frozen KNN Knowledge Base...")

    artifact = joblib.load(ARTIFACT_PATH)

    reference = pd.read_csv(
        REFERENCE_PATH,
        low_memory=False
    )

    demo = read_partition(
        SPLITS_DIR / "unknown_demo_kb_updates.parquet"
    )

    audit = pd.read_csv(AUDIT_PATH)

    if len(reference) != artifact["count"]:
        raise ValueError("KB reference count mismatch")

    # Use a real demonstration sample.
    if "demo_row" not in audit.columns or "matched" not in audit.columns:
        raise ValueError("Audit is missing demo_row or matched")

    unmatched = audit.loc[
        audit["matched"].astype(str).str.lower().eq("false")
    ]

    if unmatched.empty:
        raise ValueError("No KB-unmatched sample found in audit")

    row_index = int(unmatched.iloc[0]["demo_row"])
    flow = demo.iloc[[row_index]]

    result = explain_kb_match(
        flow=flow,
        artifact=artifact,
        reference=reference,
        top_n=5
    )

    print("\n========== KB EXPLANATION ==========")
    print("Demo row:", row_index)
    print("KB matched:", result["kb_matched"])
    print("Route:", result["route"])
    print("Nearest pattern:", result["nearest_pattern_id"])
    print("Distance:", result["nearest_distance"])
    print("Cutoff:", result["match_cutoff"])

    print("\nLargest feature differences:")

    for item in result["top_feature_differences"]:
        print(
            f"{item['feature']}: "
            f"{item['squared_difference']:.6f} "
            f"({item['distance_share']:.2%})"
        )

    # Verify against the teammate's audit.
    if "demo_row" in audit.columns:
        matching_rows = audit.loc[
            audit["demo_row"] == row_index
        ]

        if not matching_rows.empty:
            audit_row = matching_rows.iloc[0]

            assert result["kb_matched"] == bool(
                audit_row["matched"]
            )

            assert np.isclose(
                result["nearest_distance"],
                float(audit_row["nearest_distance"]),
                rtol=1e-5,
                atol=1e-7
            )

            print("\nAudit consistency: PASSED")

    print("====================================")


if __name__ == "__main__":
    main()
