
"""
Verify real historical comparison integration.

Tests:
- Real XGBoost prediction.
- Real historical nearest-neighbour comparison.
- Detection status unchanged.
- Historical evidence in alert.
- JSON serialization.
"""

from pathlib import Path

import json
import pandas as pd

from src.pipeline.integration import GC2Pipeline
from src.pipeline.alert import alert_to_dict


ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "Data_V3_SSH_Unknown_final"
    / "known_test.parquet"
)


def main():

    print("Loading test record...")

    test = pd.read_parquet(TEST_PATH)

    # Previously verified DDoS test record.
    flow = test.iloc[[4]].copy()

    pipeline = GC2Pipeline()

    alert = pipeline.predict_one(
        flow=flow,
        alert_id="GC2-HISTORICAL-TEST",
        asset_id="A001"
    )

    historical = alert.evidence.get(
        "historical_comparison"
    )

    assert historical is not None

    assert historical["nearest_historical_attack"]

    assert historical["distance"] >= 0

    assert historical["reference_count"] == 801

    assert alert.detection_status == "KNOWN_ATTACK"

    assert alert.predicted_class == "DDoS"

    # Verify JSON serialization.
    serialized = alert_to_dict(alert)

    assert (
        "historical_comparison"
        in serialized["evidence"]
    )

    json.dumps(
        serialized,
        allow_nan=False
    )

    print("\n========== HISTORICAL INTEGRATION ==========")

    print(
        "Detection:",
        alert.detection_status
    )

    print(
        "XGBoost prediction:",
        alert.predicted_class
    )

    print(
        "Nearest historical attack:",
        historical["nearest_historical_attack"]
    )

    print(
        "Historical distance:",
        historical["distance"]
    )

    print(
        "Reference count:",
        historical["reference_count"]
    )

    print("\nHISTORICAL INTEGRATION: PASSED")


if __name__ == "__main__":
    main()
