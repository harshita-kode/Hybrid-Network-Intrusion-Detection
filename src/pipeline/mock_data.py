from src.schemas.results import (
    ClassifierResult,
    HistoricalResult,
    UnknownDetectionResult,
)


# ---------------------------------------------------------
# Scenario 1: High-confidence benign traffic
# ---------------------------------------------------------

BENIGN_CLASSIFIER = ClassifierResult(
    predicted_class="BENIGN",
    confidence=0.97,
    probabilities={
        "BENIGN": 0.97,
        "DDoS": 0.02,
        "PortScan": 0.01,
    },
)


# ---------------------------------------------------------
# Scenario 2: High-confidence known attack
# ---------------------------------------------------------

KNOWN_ATTACK_CLASSIFIER = ClassifierResult(
    predicted_class="DDoS",
    confidence=0.94,
    probabilities={
        "BENIGN": 0.01,
        "DDoS": 0.94,
        "PortScan": 0.05,
    },
)

KNOWN_ATTACK_HISTORICAL = HistoricalResult(
    closest_attack="DDoS",
    similarity=0.90,
)


# ---------------------------------------------------------
# Low-confidence classifier result used by unknown cases
# ---------------------------------------------------------

LOW_CONFIDENCE_CLASSIFIER = ClassifierResult(
    predicted_class="DDoS",
    confidence=0.42,
    probabilities={
        "BENIGN": 0.33,
        "DDoS": 0.42,
        "PortScan": 0.25,
    },
)


# ---------------------------------------------------------
# Scenario 3: Previously learned unknown
# ---------------------------------------------------------

PREVIOUS_UNKNOWN_RESULT = UnknownDetectionResult(
    kb_match=True,
    matched_pattern="UNKNOWN_PATTERN_001",
    kb_match_strength=0.92,
)

PREVIOUS_UNKNOWN_HISTORICAL = HistoricalResult(
    closest_attack="PortScan",
    similarity=0.75,
)


# ---------------------------------------------------------
# Scenario 4: Potential new unknown
# ---------------------------------------------------------

NEW_UNKNOWN_RESULT = UnknownDetectionResult(
    kb_match=False,
    reconstruction_error=0.090,
    anomaly_threshold=0.040,
    anomaly_score=0.90,
)

NEW_UNKNOWN_HISTORICAL = HistoricalResult(
    closest_attack="DDoS",
    similarity=0.72,
)


# ---------------------------------------------------------
# Scenario 5: Uncertain traffic
# ---------------------------------------------------------

UNCERTAIN_RESULT = UnknownDetectionResult(
    kb_match=False,
    reconstruction_error=0.035,
    anomaly_threshold=0.040,
    anomaly_score=0.30,
)

UNCERTAIN_HISTORICAL = HistoricalResult(
    closest_attack="PortScan",
    similarity=0.45,
)