
"""
Verify historical attack-profile comparison.

Tests real records and checks deterministic nearest
neighbour results.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.historical_similarity import (
    HistoricalSimilarity
)


ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    ROOT / "data/processed"
    / "Data_V3_SSH_Unknown_final"
    / "known_test.parquet"
)


def main():

    print("Loading historical comparison...")

    comparator = HistoricalSimilarity()

    print("Loading test data...")

    test = pd.read_parquet(TEST_PATH)

    # Use known DDoS test record from previous stages.
    sample = test.iloc[[4]]

    first = comparator.compare(sample)
    second = comparator.compare(sample)

    assert first == second

    assert np.isfinite(first["distance"])
    assert first["distance"] >= 0

    assert first["nearest_historical_attack"] != "BENIGN"

    print("\n========== HISTORICAL COMPARISON ==========")
    print("Actual label:", sample.iloc[0]["Label"])
    print(
        "Nearest historical attack:",
        first["nearest_historical_attack"]
    )
    print("Distance:", first["distance"])
    print("Method:", first["method"])
    print("Reference records:", first["reference_count"])

    print("\nHISTORICAL SIMILARITY TEST: PASSED")


if __name__ == "__main__":
    main()
