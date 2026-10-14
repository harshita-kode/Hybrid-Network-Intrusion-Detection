
"""Asset-based inherent risk assessment for GC2."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AssetRiskResult:
    asset_name: str
    threat_name: str
    confidentiality: int
    integrity: int
    availability: int
    probability: int
    impact: int
    vulnerability: int
    asset_value: float
    threat_value: int
    inherent_risk: float
    acceptable_risk: Optional[float]
    decision: str
    risk_level: str = "NOT_ASSESSED"
    destination_port: Optional[int] = None
    service: Optional[str] = None
    detection_confidence: Optional[float] = None


def classify_risk_level(score: float) -> str:
    """Provisional risk severity bands."""

    if not 1 <= score <= 192:
        raise ValueError("Risk score must be between 1 and 192")

    if score <= 48:
        return "LOW"
    if score <= 96:
        return "MEDIUM"
    if score <= 144:
        return "HIGH"
    return "CRITICAL"


def calculate_asset_risk(
    asset_name: str,
    threat_name: str,
    confidentiality: int,
    integrity: int,
    availability: int,
    probability: int,
    impact: int,
    vulnerability: int,
    acceptable_risk: Optional[float] = None,
    destination_port: Optional[int] = None,
    service: Optional[str] = None,
    detection_confidence: Optional[float] = None,
) -> AssetRiskResult:

    ratings = {
        "confidentiality": (confidentiality, 1, 4),
        "integrity": (integrity, 1, 4),
        "availability": (availability, 1, 4),
        "probability": (probability, 1, 3),
        "impact": (impact, 1, 4),
        "vulnerability": (vulnerability, 1, 4),
    }

    for name, (value, minimum, maximum) in ratings.items():
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")

        if not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be between {minimum} and {maximum}"
            )

    if not asset_name.strip() or not threat_name.strip():
        raise ValueError("Asset and threat names are required")

    if acceptable_risk is not None:
        if (
            isinstance(acceptable_risk, bool)
            or not isinstance(acceptable_risk, (int, float))
            or not 0 <= acceptable_risk <= 192
        ):
            raise ValueError(
                "acceptable_risk must be between 0 and 192"
            )

    if detection_confidence is not None:
        if not 0 <= detection_confidence <= 1:
            raise ValueError(
                "detection_confidence must be between 0 and 1"
            )

    if destination_port is not None:
        if (
            type(destination_port) is not int
            or not 0 <= destination_port <= 65535
        ):
            raise ValueError("Invalid destination port")

    asset_value = (
        confidentiality + integrity + availability
    ) / 3

    threat_value = probability * impact

    inherent_risk = (
        asset_value * threat_value * vulnerability
    )

    if acceptable_risk is None:
        decision = "NOT_EVALUATED"
    elif inherent_risk <= acceptable_risk:
        decision = "ACCEPTABLE"
    else:
        decision = "REQUIRES_TREATMENT"

    return AssetRiskResult(
        asset_name=asset_name,
        threat_name=threat_name,
        confidentiality=confidentiality,
        integrity=integrity,
        availability=availability,
        probability=probability,
        impact=impact,
        vulnerability=vulnerability,
        asset_value=asset_value,
        threat_value=threat_value,
        inherent_risk=inherent_risk,
        acceptable_risk=acceptable_risk,
        decision=decision,
        destination_port=destination_port,
        service=service,
        detection_confidence=detection_confidence,
        risk_level=classify_risk_level(inherent_risk),
    )
