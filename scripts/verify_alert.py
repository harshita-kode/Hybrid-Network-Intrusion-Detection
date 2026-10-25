
from src.pipeline.alert import generate_alert


def should_reject(status, evidence):
    try:
        generate_alert(
            alert_id="GC2-INVALID",
            detection_status=status,
            evidence=evidence
        )
    except (ValueError, TypeError):
        return

    raise AssertionError(
        f"Contradictory alert accepted: {status}, {evidence}"
    )


def should_accept(status, evidence):
    alert = generate_alert(
        alert_id="GC2-VALID",
        detection_status=status,
        evidence=evidence
    )
    assert alert.detection_status == status


def main():
    should_accept(
        "KNOWN_ATTACK",
        {"classifier": "XGBoost"}
    )

    should_accept(
        "PREVIOUSLY_LEARNED_UNKNOWN",
        {"kb_matched": True}
    )

    should_accept(
        "POTENTIAL_NEW_UNKNOWN",
        {
            "kb_matched": False,
            "ae_anomalous": True
        }
    )

    should_accept(
        "UNCERTAIN",
        {
            "kb_matched": False,
            "ae_anomalous": False
        }
    )
    should_accept(
        "UNCERTAIN",
        {
            "kb_matched": True,
            "ae_called": False,
            "kb_updated": False
        }
    )

    should_reject(
        "POTENTIAL_NEW_UNKNOWN",
        {
            "kb_matched": True,
            "ae_anomalous": False
        }
    )

    should_reject(
        "PREVIOUSLY_LEARNED_UNKNOWN",
        {"kb_matched": False}
    )

    should_reject(
        "UNCERTAIN",
        {"ae_anomalous": True}
    )

    should_reject(
        "POTENTIAL_NEW_UNKNOWN",
        {
            "kb_matched": False,
            "ae_anomalous": False,
            "kb_updated": True
        }
    )

    should_reject(
        "PREVIOUSLY_LEARNED_UNKNOWN",
        {
            "kb_matched": True,
            "ae_called": True
        }
    )

    print("Valid alert scenarios: PASSED")
    print("Contradictory evidence rejection: PASSED")
    print("ALERT CONSISTENCY: PASSED")


if __name__ == "__main__":
    main()
