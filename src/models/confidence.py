import numpy as np
from sklearn.metrics import precision_score


DEFAULT_CONFIDENCE_THRESHOLD = 0.95


def get_confidence(probabilities):
    """
    Confidence is the highest predicted class probability.
    """
    return np.max(probabilities, axis=1)


def route_by_confidence(
    predictions,
    probabilities,
    threshold=DEFAULT_CONFIDENCE_THRESHOLD
):
    """
    Route predictions based on confidence.

    HIGH_CONFIDENCE:
        confidence >= threshold

    LOW_CONFIDENCE:
        confidence < threshold
    """

    confidence = get_confidence(probabilities)

    status = np.where(
        confidence >= threshold,
        "HIGH_CONFIDENCE",
        "LOW_CONFIDENCE"
    )

    return confidence, status


def find_confidence_threshold(
    y_true,
    predictions,
    probabilities,
    target_precision=0.95
):
    """
    Find the lowest confidence threshold that achieves
    the requested precision on accepted predictions.
    """

    confidence = get_confidence(probabilities)

    thresholds = np.arange(
        0.50,
        1.00,
        0.01
    )

    best_threshold = DEFAULT_CONFIDENCE_THRESHOLD

    for threshold in thresholds:

        accepted = confidence >= threshold

        if accepted.sum() == 0:
            continue

        accepted_true = np.asarray(y_true)[accepted]
        accepted_predictions = np.asarray(predictions)[accepted]

        precision = precision_score(
            accepted_true,
            accepted_predictions,
            average="weighted",
            zero_division=0
        )

        if precision >= target_precision:
            best_threshold = threshold
            break

    return best_threshold