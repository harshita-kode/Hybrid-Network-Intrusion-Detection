
"""
GC2 analyst-facing security alert generation.

Combines:
- Detection status and evidence
- XGBoost / KNN / Autoencoder explanations
- Asset-based inherent risk assessment

Does not execute models or assign risk ratings.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json
import math

from src.schemas.results import ExplanationResult
from src.analysis.risk import AssetRiskResult


VALID_STATUSES = {
    "BENIGN",
    "KNOWN_ATTACK",
    "PREVIOUSLY_LEARNED_UNKNOWN",
    "POTENTIAL_NEW_UNKNOWN",
    "UNCERTAIN",
}

VALID_METHODS = {
    "shap",
    "knn_feature_distance",
    "autoencoder_reconstruction",
}


@dataclass
class SecurityAlert:
    alert_id: str
    timestamp: str
    detection_status: str
    predicted_class: Optional[str] = None
    detection_confidence: Optional[float] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    explanations: List[ExplanationResult] = field(
        default_factory=list
    )
    risk: Optional[AssetRiskResult] = None
    message: str = ""


def _json_safe(value):
    """Convert dataclasses and numeric values to JSON-safe data."""

    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "item"):
        return _json_safe(value.item())

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite value in alert")
        return value

    raise TypeError(
        f"Unsupported alert value type: {type(value).__name__}"
    )


def validate_detection_consistency(status, evidence):
    """
    Reject contradictory routing evidence.

    Missing evidence is allowed at this stage.
    Complete evidence requirements will be checked
    separately during full pipeline integration.
    """

    def optional_bool(name):
        value = evidence.get(name)

        if value is not None and type(value) is not bool:
            raise TypeError(
                f"{name} must be a boolean or None"
            )

        return value

    kb_matched = optional_bool("kb_matched")
    ae_anomalous = optional_bool("ae_anomalous")
    ae_called = optional_bool("ae_called")
    kb_updated = optional_bool("kb_updated")

    if kb_matched is True and ae_called is True:
        raise ValueError(
            "Autoencoder must not run after a KB match"
        )

    if kb_matched is True and ae_anomalous is not None:
        raise ValueError(
            "Autoencoder result provided for KB-matched flow"
        )

    if kb_matched is True and kb_updated is True:
        raise ValueError(
            "A KB-matched flow should not trigger new insertion"
        )

    if kb_updated is True and (
        kb_matched is not False
        or ae_anomalous is not True
    ):
        raise ValueError(
            "KB insertion requires KB miss and AE anomaly"
        )

    if status == "PREVIOUSLY_LEARNED_UNKNOWN":
        if kb_matched is False:
            raise ValueError(
                "Previously learned unknown requires KB match"
            )

    if status == "POTENTIAL_NEW_UNKNOWN":
        if kb_matched is True or ae_anomalous is False:
            raise ValueError(
                "Potential new unknown requires KB miss "
                "and positive anomaly evidence"
            )

    
    if status == "UNCERTAIN":

        # A KB match may still require investigation
        # when the stored pattern has no verified label.
        #
        # However, a confirmed Autoencoder anomaly
        # after a KB miss should use
        # POTENTIAL_NEW_UNKNOWN instead.

        if ae_anomalous is True:
            raise ValueError(
                "Uncertain status conflicts with "
                "a positive Autoencoder anomaly"
            )



def generate_alert(
    alert_id: str,
    detection_status: str,
    predicted_class: Optional[str] = None,
    detection_confidence: Optional[float] = None,
    evidence: Optional[Dict[str, Any]] = None,
    explanations: Optional[List[ExplanationResult]] = None,
    risk: Optional[AssetRiskResult] = None,
) -> SecurityAlert:
    """Create a structured GC2 alert from existing component outputs."""

    if not isinstance(alert_id, str) or not alert_id.strip():
        raise ValueError("A nonempty alert_id is required")

    if detection_status not in VALID_STATUSES:
        raise ValueError(
            f"Unsupported detection status: {detection_status}"
        )

    if detection_confidence is not None:
        if (
            isinstance(detection_confidence, bool)
            or not isinstance(detection_confidence, (int, float))
            or not math.isfinite(detection_confidence)
            or not 0 <= detection_confidence <= 1
        ):
            raise ValueError(
                "Detection confidence must be between 0 and 1"
            )

    if evidence is None:
        evidence = {}

    if not isinstance(evidence, dict):
        raise TypeError("evidence must be a dictionary")

    validate_detection_consistency(
        detection_status,
        evidence
    )

    if explanations is None:
        explanations = []

    if not isinstance(explanations, list):
        raise TypeError("explanations must be a list")

    for explanation in explanations:
        if not isinstance(explanation, ExplanationResult):
            raise TypeError(
                "Each explanation must be an ExplanationResult"
            )

        if explanation.method not in VALID_METHODS:
            raise ValueError(
                f"Unknown explanation method: {explanation.method}"
            )

    if risk is not None and not isinstance(risk, AssetRiskResult):
        raise TypeError("risk must be an AssetRiskResult")

    messages = {
        "BENIGN": "Traffic classified as benign.",
        "KNOWN_ATTACK": "Known attack classification reported.",
        "PREVIOUSLY_LEARNED_UNKNOWN":
            "Traffic matched a previously stored KB pattern.",
        "POTENTIAL_NEW_UNKNOWN":
            "Anomalous traffic without a sufficient KB match.",
        "UNCERTAIN":
            "Traffic requires further investigation.",
    }

    alert = SecurityAlert(
        alert_id=alert_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        detection_status=detection_status,
        predicted_class=predicted_class,
        detection_confidence=(
            float(detection_confidence)
            if detection_confidence is not None
            else None
        ),
        evidence=dict(evidence),
        explanations=list(explanations),
        risk=risk,
        message=messages[detection_status],
    )

    # Verify that all supplied information is serializable.
    _json_safe(alert)

    return alert



def alert_to_dict(alert: SecurityAlert) -> Dict[str, Any]:
    """
    Convert a security alert into a clean JSON-compatible
    dictionary without repeating explanation features.
    """

    if not isinstance(alert, SecurityAlert):
        raise TypeError("Expected SecurityAlert")

    # Convert the existing alert to a dictionary.
    result = _json_safe(alert)

    cleaned_explanations = []

    for explanation in result["explanations"]:

        method = explanation["method"]

        cleaned = {
            "method": method
        }

        if method == "shap":

            # SHAP contributions already contain the
            # feature names, feature values and SHAP values.
            cleaned["contributions"] = explanation[
                "contributions"
            ]

        else:

            # KNN and Autoencoder use feature_details.
            cleaned["feature_details"] = explanation[
                "feature_details"
            ]

        # Preserve model-specific supporting information.
        if explanation.get("details"):
            cleaned["details"] = explanation["details"]

        cleaned_explanations.append(cleaned)

    result["explanations"] = cleaned_explanations

    return result


def alert_to_json(alert: SecurityAlert) -> str:
    """Generate formatted JSON for storage or integration."""
    return json.dumps(
        alert_to_dict(alert),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )


def format_alert(alert: SecurityAlert) -> str:
    """Create a readable analyst-facing alert."""

    lines = [
        "========== SECURITY ALERT ==========",
        f"Alert ID: {alert.alert_id}",
        f"Timestamp: {alert.timestamp}",
        f"Detection status: {alert.detection_status}",
        f"Message: {alert.message}",
    ]

    if alert.predicted_class is not None:
        lines.append(
            f"Classifier prediction: {alert.predicted_class}"
        )

    if alert.detection_confidence is not None:
        lines.append(
            f"Detection confidence: "
            f"{alert.detection_confidence:.2%}"
        )

    lines.append("\n--- DETECTION EVIDENCE ---")

    if alert.evidence:
            
        for key, value in alert.evidence.items():

            if key in (
                "historical_comparison",
                "historical_comparison_role",
            ):
                continue

            lines.append(f"{key}: {value}")

    else:
        lines.append("No additional evidence provided.")

    lines.append("\n--- EXPLAINABILITY ---")

    if not alert.explanations:
        lines.append("No explanation available.")
    else:
        for explanation in alert.explanations:
            lines.append(
                f"\nMethod: {explanation.method}"
            )

            if explanation.contributions:
                lines.append("SHAP contributions:")

                for item in explanation.contributions:
                    lines.append(
                        f"  {item.feature}: "
                        f"{item.shap_value:+.6f}"
                    )

            if (
                explanation.feature_details
                and explanation.method != "shap"
            ):
                lines.append("Feature evidence:")

                for item in explanation.feature_details:
                    lines.append(
                        f"  {item.get('feature')}: "
                        f"{item}"
                    )

            if explanation.details:
                lines.append("Supporting details:")

                for key, value in explanation.details.items():
                    lines.append(f"  {key}: {value}")

    
    # ==================================================
    # HISTORICAL ATTACK COMPARISON
    # ==================================================

    historical = alert.evidence.get(
        "historical_comparison"
    )

    lines.append(
        "\n--- HISTORICAL ATTACK COMPARISON ---"
    )

    if historical is None:

        lines.append(
            "Historical comparison unavailable."
        )

    else:

        lines.append(
            "Nearest historical attack: "
            f"{historical['nearest_historical_attack']}"
        )

        lines.append(
            "Distance: "
            f"{historical['distance']:.6f}"
        )

        lines.append(
            "Method: "
            f"{historical['method']}"
        )

        lines.append(
            "Reference records: "
            f"{historical['reference_count']}"
        )

        lines.append(
            "Interpretation: "
            f"{historical['interpretation']}"
        )

        lines.append(
            "Role: SUPPORTING_EVIDENCE_ONLY"
        )


    lines.append("\n--- ASSET-BASED RISK ---")

    if alert.risk is None:
        lines.extend([
            "Risk score: NOT_AVAILABLE",
            "Risk level: NOT_ASSESSED",
            "Risk decision: NOT_AVAILABLE",
            "Risk basis: Insufficient validated asset/threat ratings",
        ])
    else:
        r = alert.risk

        lines.extend([
            f"Asset: {r.asset_name}",
            f"Threat: {r.threat_name}",
            f"Asset value: {r.asset_value:.2f}",
            f"Threat likelihood rating: {r.probability}",
            f"Impact rating: {r.impact}",
            f"Threat value: {r.threat_value}",
            f"Vulnerability rating: {r.vulnerability}",
            f"Inherent risk: {r.inherent_risk:.2f}",
            f"Risk level: {r.risk_level}",
            f"Acceptable risk: {r.acceptable_risk}",
            f"Risk decision: {r.decision}",
        ])

        if r.destination_port is not None:
            lines.append(
                f"Destination port: {r.destination_port}"
            )

        if r.service is not None:
            lines.append(f"Service: {r.service}")

    lines.append("========================================")

    return "\n".join(lines)
