
"""Convert GC2 explanation outputs into a common schema."""

from src.schemas.results import ExplanationResult


def adapt_shap_explanation(explanation):
    """
    Adapt the existing XGBoost SHAP explanation.

    Preserves original SHAP feature contributions.
    """
    if not isinstance(explanation, ExplanationResult):
        raise TypeError(
            "Expected an ExplanationResult from XGBoost SHAP"
        )

    return ExplanationResult(
        method="shap",
        top_features=list(explanation.top_features),
        contributions=list(explanation.contributions),
        feature_details=[
            {
                "feature": item.feature,
                "feature_value": item.feature_value,
                "shap_value": item.shap_value,
            }
            for item in explanation.contributions
        ],
        details={
            "description": (
                "Feature contributions to the "
                "XGBoost predicted-class output"
            )
        }
    )


def adapt_kb_explanation(result):
    """
    Adapt the existing KNN nearest-neighbour explanation.

    Squared differences are measured in the
    KNN transformed feature space.
    """
    required = {
        "kb_matched",
        "nearest_pattern_id",
        "nearest_distance",
        "match_cutoff",
        "top_feature_differences",
    }

    missing = required - result.keys()

    if missing:
        raise ValueError(
            f"Missing KNN explanation fields: {sorted(missing)}"
        )

    feature_details = list(
        result["top_feature_differences"]
    )

    return ExplanationResult(
        method="knn_feature_distance",
        top_features=[
            item["feature"]
            for item in feature_details
        ],
        feature_details=feature_details,
        details={
            "kb_matched": result["kb_matched"],
            "route": result.get("route"),
            "nearest_pattern_id": result["nearest_pattern_id"],
            "nearest_distance": result["nearest_distance"],
            "match_cutoff": result["match_cutoff"],
        }
    )


def adapt_autoencoder_explanation(result):
    """
    Adapt Version20 Autoencoder reconstruction explanation.

    Feature errors are measured in the saved
    Autoencoder's transformed feature space.
    """
    required = {
        "anomaly_mse",
        "threshold",
        "anomalous",
        "top_features",
        "feature_count",
        "ensemble_size",
    }

    missing = required - result.keys()

    if missing:
        raise ValueError(
            "Missing Autoencoder explanation fields: "
            f"{sorted(missing)}"
        )

    feature_details = list(result["top_features"])

    return ExplanationResult(
        method="autoencoder_reconstruction",
        top_features=[
            item["feature"]
            for item in feature_details
        ],
        feature_details=feature_details,
        details={
            "anomaly_mse": result["anomaly_mse"],
            "threshold": result["threshold"],
            "anomalous": result["anomalous"],
            "feature_count": result["feature_count"],
            "ensemble_size": result["ensemble_size"],
            "mse_consistency": result.get(
                "mse_consistency"
            ),
        }
    )
