
"""Test GC2 security alert generation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schemas.results import (
    ExplanationResult,
    FeatureContribution,
)

from src.analysis.explanation_adapter import (
    adapt_shap_explanation,
    adapt_kb_explanation,
    adapt_autoencoder_explanation,
)

from src.analysis.risk import calculate_asset_risk

from src.pipeline.alert import (
    generate_alert,
    format_alert,
    alert_to_json,
)


def test_known_attack():
    """Known DDoS prediction with SHAP and risk."""

    shap = adapt_shap_explanation(
        ExplanationResult(
            top_features=[
                "Bwd Packet Length Max",
                "Fwd RST Flags",
            ],
            contributions=[
                FeatureContribution(
                    feature="Bwd Packet Length Max",
                    feature_value=4380.0,
                    shap_value=3.166170,
                ),
                FeatureContribution(
                    feature="Fwd RST Flags",
                    feature_value=1.0,
                    shap_value=1.206848,
                ),
            ],
        )
    )

    # Ratings from the team's worked risk example.
    # They are not derived from this SHAP sample.
    risk = calculate_asset_risk(
        asset_name="Production Web Server",
        threat_name="DDoS",
        confidentiality=4,
        integrity=4,
        availability=4,
        probability=3,
        impact=4,
        vulnerability=3,
        destination_port=443,
        service="HTTPS",
        detection_confidence=0.96,
    )

    alert = generate_alert(
        alert_id="GC2-TEST-001",
        detection_status="KNOWN_ATTACK",
        predicted_class="DDoS",
        detection_confidence=0.96,
        evidence={
            "classifier": "XGBoost",
            "note": "Mocked integration example",
        },
        explanations=[shap],
        risk=risk,
    )

    assert alert.detection_status == "KNOWN_ATTACK"
    assert alert.risk.inherent_risk == 144
    assert alert.explanations[0].method == "shap"

    return alert


def test_kb_match():
    """Previously encountered KB pattern."""

    kb = adapt_kb_explanation({
        "kb_matched": True,
        "route": "KB_MATCH",
        "nearest_pattern_id": "KB_0000086",
        "nearest_distance": 0.0472235929,
        "match_cutoff": 0.9453825242,
        "top_feature_differences": [
            {
                "feature": "Src Port",
                "squared_difference": 0.000572,
                "distance_share": 0.2563,
            }
        ],
    })

    alert = generate_alert(
        alert_id="GC2-TEST-002",
        detection_status="PREVIOUSLY_LEARNED_UNKNOWN",
        evidence={
            "kb_matched": True,
            "nearest_pattern_id": "KB_0000086",
        },
        explanations=[kb],
    )

    assert alert.explanations[0].method == (
        "knn_feature_distance"
    )
    assert alert.risk is None

    return alert


def test_new_unknown():
    """KB miss followed by anomalous Autoencoder result."""

    kb = adapt_kb_explanation({
        "kb_matched": False,
        "route": "AUTOENCODER",
        "nearest_pattern_id": "KB_0000601",
        "nearest_distance": 1.1375812215,
        "match_cutoff": 0.9453825242,
        "top_feature_differences": [
            {
                "feature": "Bwd IAT Min",
                "squared_difference": 1.0,
                "distance_share": 0.7727,
            }
        ],
    })

    ae = adapt_autoencoder_explanation({
        "anomaly_mse": 0.5278878063,
        "threshold": 0.0467876471,
        "anomalous": True,
        "feature_count": 93,
        "ensemble_size": 3,
        "mse_consistency": True,
        "top_features": [
            {
                "feature": "Fwd Bytes/Bulk Avg",
                "squared_error": 11.88644227,
                "error_share": 0.2421,
            },
            {
                "feature": "src_ssh_connections_60s",
                "squared_error": 3.10565539,
                "error_share": 0.0633,
            },
        ],
    })

    alert = generate_alert(
        alert_id="GC2-TEST-003",
        detection_status="POTENTIAL_NEW_UNKNOWN",
        evidence={
            "kb_matched": False,
            "ae_anomalous": True,
            "kb_updated": False,
            "note": "Explanation test; no KB insertion performed",
        },
        explanations=[kb, ae],
    )

    assert len(alert.explanations) == 2
    assert alert.risk is None

    return alert


def main():
    alerts = [
        test_known_attack(),
        test_kb_match(),
        test_new_unknown(),
    ]

    for alert in alerts:
        print("\n" + format_alert(alert))
        print("\nJSON serialization: ", end="")
        json_output = alert_to_json(alert)

        assert '"alert_id"' in json_output
        print("PASSED")

    print("\nAll security alert tests passed.")


if __name__ == "__main__":
    main()
