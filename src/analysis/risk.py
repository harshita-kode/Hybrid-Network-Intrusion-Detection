from src.schemas.results import RiskResult


# Prototype risk boundaries.
# These are NOT final experimental thresholds.
LOW_MAX = 39.0
MEDIUM_MAX = 69.0


def risk_level_from_score(score: float) -> str:
    """
    Convert a numerical risk score (0-100) into a risk level.

    Current boundaries are provisional and will be refined
    after real model outputs are available.
    """

    score = max(0.0, min(100.0, score))

    if score <= LOW_MAX:
        return "LOW"

    if score <= MEDIUM_MAX:
        return "MEDIUM"

    return "HIGH"


def build_risk_result(score: float) -> RiskResult:
    """
    Build a standardized risk result.

    The actual risk scoring formula is intentionally not
    finalized at this stage.
    """

    score = max(0.0, min(100.0, score))

    return RiskResult(
        score=round(score, 2),
        level=risk_level_from_score(score),
    )