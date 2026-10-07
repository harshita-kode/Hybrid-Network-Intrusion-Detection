from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ============================================================
# CLASSIFIER RESULT
# ============================================================

@dataclass
class ClassifierResult:
    """
    Output produced by the known-attack classifier.
    """

    predicted_class: str
    confidence: float
    probabilities: Dict[str, float] = field(
        default_factory=dict
    )


# ============================================================
# UNKNOWN DETECTION RESULT
# ============================================================

@dataclass
class UnknownDetectionResult:
    """
    Output produced by the knowledge-base and
    anomaly-detection stages.
    """

    kb_match: bool

    matched_pattern: Optional[str] = None
    kb_match_strength: Optional[float] = None

    reconstruction_error: Optional[float] = None
    anomaly_threshold: Optional[float] = None
    anomaly_score: Optional[float] = None


# ============================================================
# HISTORICAL COMPARISON RESULT
# ============================================================

@dataclass
class HistoricalResult:
    """
    Output produced by historical attack comparison.
    """

    closest_attack: Optional[str] = None
    similarity: Optional[float] = None


# ============================================================
# SHAP EXPLANATION RESULT
# ============================================================

@dataclass
class FeatureContribution:
    """
    SHAP contribution of one network-flow feature.
    """

    feature: str
    feature_value: float
    shap_value: float


@dataclass
class ExplanationResult:
    """
    Explainability information for a prediction.
    """

    top_features: List[str] = field(
        default_factory=list
    )

    contributions: List[FeatureContribution] = field(
        default_factory=list
    )


# ============================================================
# RISK ASSESSMENT RESULT
# ============================================================

@dataclass
class RiskResult:
    """
    Final numerical risk assessment.
    """

    score: Optional[float] = None
    level: Optional[str] = None


# ============================================================
# COMPLETE DETECTION RESULT
# ============================================================

@dataclass
class DetectionResult:
    """
    Combined result produced by the complete NIDS pipeline.
    """

    status: str
    classifier: ClassifierResult

    unknown_detection: Optional[
        UnknownDetectionResult
    ] = None

    historical: Optional[
        HistoricalResult
    ] = None

    explanation: Optional[
        ExplanationResult
    ] = None

    risk: Optional[
        RiskResult
    ] = None