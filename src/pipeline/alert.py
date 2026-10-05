from src.schemas.results import DetectionResult


def generate_alert(result: DetectionResult) -> str:
    lines = [
        "=" * 50,
        "HYBRID NETWORK INTRUSION DETECTION",
        "=" * 50,
        f"Detection Status : {result.status}",
        f"Predicted Class  : {result.classifier.predicted_class}",
        f"Confidence       : {result.classifier.confidence:.2%}",
    ]

    if result.unknown_detection is not None:
        lines.append(
            f"Knowledge Base   : "
            f"{'MATCH' if result.unknown_detection.kb_match else 'NO MATCH'}"
        )

        if result.unknown_detection.reconstruction_error is not None:
            lines.append(
                f"Reconstruction Error : "
                f"{result.unknown_detection.reconstruction_error:.4f}"
            )

        if result.unknown_detection.anomaly_threshold is not None:
            lines.append(
                f"Anomaly Threshold    : "
                f"{result.unknown_detection.anomaly_threshold:.4f}"
            )

    if result.historical is not None:
        if result.historical.closest_attack is not None:
            lines.append(
                f"Historical Match : {result.historical.closest_attack}"
            )

        if result.historical.similarity is not None:
            lines.append(
                f"Similarity       : {result.historical.similarity:.2%}"
            )

    if result.risk is not None:
        if result.risk.score is not None:
            lines.append(f"Risk Score       : {result.risk.score:.2f}/100")

        if result.risk.level is not None:
            lines.append(f"Risk Level       : {result.risk.level}")

    if result.explanation is not None and result.explanation.top_features:
        lines.append("Top Features     :")

        for feature in result.explanation.top_features:
            lines.append(f"  - {feature}")

    lines.append("=" * 50)

    return "\n".join(lines)