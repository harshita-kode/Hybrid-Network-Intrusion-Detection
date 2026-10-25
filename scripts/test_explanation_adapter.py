
"""Test the unified explanation adapters with sample outputs."""

from dataclasses import asdict

from src.schemas.results import (
    ExplanationResult,
    FeatureContribution,
)

from src.analysis.explanation_adapter import (
    adapt_shap_explanation,
    adapt_kb_explanation,
    adapt_autoencoder_explanation,
)


def main():

    # --------------------------------------------------
    # XGBOOST SHAP
    # --------------------------------------------------

    shap_result = ExplanationResult(
        top_features=["Bwd Packet Length Max"],
        contributions=[
            FeatureContribution(
                feature="Bwd Packet Length Max",
                feature_value=4380.0,
                shap_value=3.166170
            )
        ]
    )

    unified_shap = adapt_shap_explanation(
        shap_result
    )

    assert unified_shap.method == "shap"
    assert unified_shap.top_features == [
        "Bwd Packet Length Max"
    ]

    # --------------------------------------------------
    # KNN KNOWLEDGE BASE
    # --------------------------------------------------

    kb_result = {
        "kb_matched": False,
        "route": "AUTOENCODER",
        "nearest_pattern_id": "KB_0000601",
        "nearest_distance": 1.1375812215,
        "match_cutoff": 0.9453825242,
        "top_feature_differences": [
            {
                "feature": "Bwd IAT Min",
                "squared_difference": 1.0,
                "distance_share": 0.7727
            }
        ]
    }

    unified_kb = adapt_kb_explanation(
        kb_result
    )

    assert unified_kb.method == "knn_feature_distance"
    assert unified_kb.details["kb_matched"] is False

    # --------------------------------------------------
    # AUTOENCODER
    # --------------------------------------------------

    ae_result = {
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
                "error_share": 0.2421
            }
        ]
    }

    unified_ae = adapt_autoencoder_explanation(
        ae_result
    )

    assert (
        unified_ae.method
        == "autoencoder_reconstruction"
    )
    assert unified_ae.details["anomalous"] is True

    # --------------------------------------------------
    # VERIFY SERIALIZATION
    # --------------------------------------------------

    for explanation in [
        unified_shap,
        unified_kb,
        unified_ae
    ]:
        serialized = asdict(explanation)

        assert isinstance(serialized, dict)
        assert "method" in serialized
        assert "top_features" in serialized

        print(
            explanation.method,
            "-> PASSED"
        )

    print(
        "\nAll unified explanation tests passed."
    )


if __name__ == "__main__":
    main()
