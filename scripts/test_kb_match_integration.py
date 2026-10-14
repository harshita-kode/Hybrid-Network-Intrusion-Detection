
"""
Controlled GC2 KB-match integration test.

Uses a real low-confidence XGBoost flow and real SHAP,
but injects a simulated KB-match explanation.

Does not change model thresholds or KB artifacts.
"""

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.integration import GC2Pipeline
from src.pipeline.alert import (
    format_alert,
    alert_to_json,
)

import pandas as pd


TEST_PATH = (
    ROOT / "data"
    / "incoming"
    / "low_confidence_traffic.csv"
)


def simulated_kb_match(*args, **kwargs):
    """
    Return a controlled KB-match result matching
    the existing explanation adapter's schema.
    """
    return {
        "kb_matched": True,
        "route": "KB_MATCH",
        "nearest_pattern_id": "KB_TEST_MATCH",
        "nearest_distance": 0.1,
        "match_cutoff": 0.9453825241925351,
        "top_feature_differences": [
            {
                "feature": "Src Port",
                "squared_difference": 0.01,
                "distance_share": 1.0,
            }
        ],
    }


def main():
    print("Loading real low-confidence flow...")

    data = pd.read_csv(TEST_PATH)

    if data.empty:
        raise ValueError("Test dataset is empty")

    flow = data.iloc[[0]]

    pipeline = GC2Pipeline()

    print("Running controlled KB-match test...")

    with patch(
        "src.pipeline.integration.explain_kb_match",
        side_effect=simulated_kb_match
    ) as mocked_kb:

        alert = pipeline.predict_one(
            flow=flow,
            alert_id="GC2-KB-MATCH-TEST",
        )

    # Ensure the real pipeline called the KB.
    assert mocked_kb.call_count == 1

    # Verify correct routing.
    assert alert.evidence["confidence_route"] == (
        "LOW_CONFIDENCE"
    )

    assert alert.evidence["kb_called"] is True
    assert alert.evidence["kb_matched"] is True
    assert alert.evidence["kb_route"] == "KB_MATCH"

    # Autoencoder must be skipped.
    assert alert.evidence["ae_called"] is False

    # KB match is not automatically confirmed malicious.
    assert alert.detection_status == "UNCERTAIN"

    # Both SHAP and KNN explanation should be present.
    methods = [
        explanation.method
        for explanation in alert.explanations
    ]

    assert methods == [
        "shap",
        "knn_feature_distance",
    ]

    # JSON serialization must work.
    json_output = alert_to_json(alert)
    assert '"kb_matched": true' in json_output

    print("\n" + format_alert(alert))

    print("\nKB-MATCH ROUTING: PASSED")
    print("AUTOENCODER SKIPPED: PASSED")
    print("EXPLANATION INTEGRATION: PASSED")
    print("JSON SERIALIZATION: PASSED")
    print("\nALL CONTROLLED KB-MATCH TESTS PASSED")


if __name__ == "__main__":
    main()
