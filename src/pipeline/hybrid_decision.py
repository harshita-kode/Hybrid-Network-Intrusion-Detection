
"""GC2 shadow hybrid decision logic.

Combines existing classifier, frozen KB and Autoencoder
evidence without modifying the original production decision.
"""


def resolve_hybrid_decision(
    production_alert,
    kb_result,
    ae_result=None,
):
    """
    Return an evidence-based shadow decision.

    This does not promote KB candidates, update models,
    or modify the production alert.
    """

    status = production_alert.detection_status
    evidence = production_alert.evidence

    original_status = status

    kb_matched = bool(
        kb_result["kb_matched"]
    )

    # Preserve accepted known attacks.
    if status == "KNOWN_ATTACK":
        return {
            "original_status": original_status,
            "shadow_status": "KNOWN_ATTACK",
            "decision_changed": False,
            "requires_review": False,
            "reason": "Accepted known attack classification",
            "kb_matched": kb_matched,
            "ae_anomalous": None,
        }

    # A KB match alone is not enough to confirm
    # malicious activity without verified provenance.
    if kb_matched:
        return {
            "original_status": original_status,
            "shadow_status": "UNCERTAIN",
            "decision_changed": status != "UNCERTAIN",
            "requires_review": True,
            "reason": (
                "Stored KB pattern matched; "
                "malicious provenance not verified"
            ),
            "kb_matched": True,
            "ae_anomalous": None,
        }

    # No Autoencoder result means the investigation
    # is incomplete.
    if ae_result is None or "not_run" in ae_result:
        return {
            "original_status": original_status,
            "shadow_status": "UNCERTAIN",
            "decision_changed": status != "UNCERTAIN",
            "requires_review": True,
            "reason": (
                "KB miss; Autoencoder evidence unavailable"
            ),
            "kb_matched": False,
            "ae_anomalous": None,
        }

    if not ae_result.get("mse_consistency", False):
        raise ValueError(
            "Autoencoder MSE consistency verification failed"
        )

    mse = float(ae_result["anomaly_mse"])
    threshold = float(ae_result["threshold"])
    anomalous = bool(ae_result["anomalous"])

    if anomalous != (mse > threshold):
        raise ValueError(
            "Autoencoder decision contradicts threshold"
        )

    if anomalous:
        new_status = "POTENTIAL_NEW_UNKNOWN"
        reason = (
            "KB miss followed by Autoencoder anomaly"
        )
        review = True
    else:
        # No anomaly does not independently establish
        # that the traffic is harmless.
        new_status = (
            "BENIGN"
            if status == "BENIGN"
            else "UNCERTAIN"
        )
        reason = (
            "KB miss; Autoencoder did not detect anomaly"
        )
        review = new_status == "UNCERTAIN"

    return {
        "original_status": original_status,
        "shadow_status": new_status,
        "decision_changed": new_status != status,
        "requires_review": review,
        "reason": reason,
        "kb_matched": False,
        "ae_anomalous": anomalous,
        "ae_mse": mse,
        "ae_threshold": threshold,
    }
