from src.analysis.risk import build_risk_result
from src.pipeline.alert import generate_alert
from src.pipeline.mock_data import (
    BENIGN_CLASSIFIER,
    KNOWN_ATTACK_CLASSIFIER,
    KNOWN_ATTACK_HISTORICAL,
    LOW_CONFIDENCE_CLASSIFIER,
    NEW_UNKNOWN_HISTORICAL,
    NEW_UNKNOWN_RESULT,
    PREVIOUS_UNKNOWN_HISTORICAL,
    PREVIOUS_UNKNOWN_RESULT,
    UNCERTAIN_HISTORICAL,
    UNCERTAIN_RESULT,
)
from src.pipeline.router import determine_detection_status
from src.schemas.results import DetectionResult


def run_scenario(
    name,
    classifier,
    mock_risk_score,
    unknown=None,
    historical=None,
):
    status = determine_detection_status(
        classifier=classifier,
        unknown=unknown,
    )

    risk = build_risk_result(mock_risk_score)

    result = DetectionResult(
        status=status,
        classifier=classifier,
        unknown_detection=unknown,
        historical=historical,
        risk=risk,
    )

    print()
    print(f"TEST SCENARIO: {name}")
    print(generate_alert(result))


def main():
    run_scenario(
        name="Benign Traffic",
        classifier=BENIGN_CLASSIFIER,
        mock_risk_score=2,
    )

    run_scenario(
        name="Known Attack",
        classifier=KNOWN_ATTACK_CLASSIFIER,
        historical=KNOWN_ATTACK_HISTORICAL,
        mock_risk_score=92,
    )

    run_scenario(
        name="Previously Learned Unknown",
        classifier=LOW_CONFIDENCE_CLASSIFIER,
        unknown=PREVIOUS_UNKNOWN_RESULT,
        historical=PREVIOUS_UNKNOWN_HISTORICAL,
        mock_risk_score=76,
    )

    run_scenario(
        name="Potential New Unknown",
        classifier=LOW_CONFIDENCE_CLASSIFIER,
        unknown=NEW_UNKNOWN_RESULT,
        historical=NEW_UNKNOWN_HISTORICAL,
        mock_risk_score=79,
    )

    run_scenario(
        name="Uncertain Traffic",
        classifier=LOW_CONFIDENCE_CLASSIFIER,
        unknown=UNCERTAIN_RESULT,
        historical=UNCERTAIN_HISTORICAL,
        mock_risk_score=31,
    )


if __name__ == "__main__":
    main()