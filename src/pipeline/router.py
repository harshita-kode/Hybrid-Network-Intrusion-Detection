from typing import Optional

from src.schemas.constants import (
    BENIGN,
    KNOWN_ATTACK,
    POTENTIAL_NEW_UNKNOWN,
    PREVIOUSLY_LEARNED_UNKNOWN,
    UNCERTAIN,
)

from src.schemas.results import (
    ClassifierResult,
    UnknownDetectionResult,
)


def determine_detection_status(
    classifier: ClassifierResult,
    unknown: Optional[UnknownDetectionResult] = None,
    confidence_threshold: float = 0.80,
) -> str:
    """
    Determine the overall detection status.

    The confidence threshold is currently a temporary integration
    value and is NOT the final experimentally selected threshold.
    """

    # High-confidence classifier prediction
    if classifier.confidence >= confidence_threshold:
        if classifier.predicted_class.upper() == "BENIGN":
            return BENIGN

        return KNOWN_ATTACK

    # Low confidence requires deeper investigation.
    if unknown is None:
        return UNCERTAIN

    # Previously learned unknown pattern.
    if unknown.kb_match:
        return PREVIOUSLY_LEARNED_UNKNOWN

    # No KB match: anomaly evidence is required.
    if (
        unknown.reconstruction_error is None
        or unknown.anomaly_threshold is None
    ):
        return UNCERTAIN

    # Autoencoder indicates anomalous behaviour.
    if unknown.reconstruction_error >= unknown.anomaly_threshold:
        return POTENTIAL_NEW_UNKNOWN

    # Low-confidence classification with insufficient anomaly evidence.
    return UNCERTAIN