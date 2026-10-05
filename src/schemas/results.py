from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ClassifierResult:
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)


@dataclass
class UnknownDetectionResult:
    kb_match: bool
    matched_pattern: Optional[str] = None
    kb_match_strength: Optional[float] = None

    reconstruction_error: Optional[float] = None
    anomaly_threshold: Optional[float] = None
    anomaly_score: Optional[float] = None


@dataclass
class HistoricalResult:
    closest_attack: Optional[str] = None
    similarity: Optional[float] = None


@dataclass
class ExplanationResult:
    top_features: List[str] = field(default_factory=list)


@dataclass
class RiskResult:
    score: Optional[float] = None
    level: Optional[str] = None


@dataclass
class DetectionResult:
    status: str

    classifier: ClassifierResult
    unknown_detection: Optional[UnknownDetectionResult] = None
    historical: Optional[HistoricalResult] = None
    explanation: Optional[ExplanationResult] = None
    risk: Optional[RiskResult] = None