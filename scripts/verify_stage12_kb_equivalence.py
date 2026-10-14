
"""Verify Stage 12B batch KB decisions against production search."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from notebooks.kb.kb_flow import search, transform

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data/processed/Data_V3_SSH_Unknown_final"
SPLIT_DIR = DATA_DIR / "unknown_kb_splits"

ARTIFACT_PATH = ROOT / "saved-models/knn_kb/initial_kb.joblib"
REFERENCE_PATH = ROOT / "data/processed/knn_kb/knowledge_base.csv"

SAMPLE_SIZE = 50


def verify_group(frame, artifact, reference, group):
    sample = frame.sample(
        n=min(SAMPLE_SIZE, len(frame)),
        random_state=42,
    )

    vectors = np.asarray(
        transform(sample, artifact),
        dtype=np.float64,
    )

    distances, _ = artifact["index"].kneighbors(
        vectors,
        n_neighbors=1,
    )

    fast_matches = (
        distances[:, 0] <= artifact["cutoff"]
    )

    actual_results = search(
        sample,
        artifact,
        reference,
        k=5,
        trace=False,
    )

    actual_matches = np.array(
        [bool(item["matched"]) for item in actual_results]
    )

    disagreements = int(
        np.count_nonzero(
            fast_matches != actual_matches
        )
    )

    print(
        f"{group}: {len(sample)} tested, "
        f"{disagreements} disagreements"
    )

    if disagreements:
        raise AssertionError(
            f"{group} KB decisions disagree"
        )


def main():
    print("Loading frozen KB...")
    artifact = joblib.load(ARTIFACT_PATH)
    reference = pd.read_csv(REFERENCE_PATH, low_memory=False)

    if len(reference) != artifact["count"]:
        raise ValueError("KB reference/index count mismatch")

    unknown = pd.read_parquet(
        SPLIT_DIR / "unknown_final_test.parquet"
    )

    known = pd.read_parquet(
        DATA_DIR / "known_test.parquet"
    )

    benign = known.loc[
        known["Label"] == "BENIGN"
    ]

    print("\n========== KB EQUIVALENCE ==========")

    verify_group(
        unknown, artifact, reference, "UNKNOWN"
    )

    verify_group(
        benign, artifact, reference, "BENIGN"
    )

    print("\nSTAGE 12C KB EQUIVALENCE: PASSED")


if __name__ == "__main__":
    main()
