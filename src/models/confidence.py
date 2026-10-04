# src/models/confidence.py

import numpy as np
from sklearn.metrics import precision_score


def get_confidence(probabilities):
    """
    Confidence = highest class probability.
    """

    return np.max(
        probabilities,
        axis=1
    )


def find_confidence_threshold(
    y_true,
    predictions,
    probabilities,
    target_precision=0.95
):
    """
    Find the lowest confidence threshold that gives
    at least the target precision on accepted predictions.

    Samples below the threshold are considered uncertain.
    """

    confidence = get_confidence(
        probabilities
    )

    thresholds = np.arange(
        0.50,
        1.00,
        0.01
    )

    best_threshold = 0.80

    for threshold in thresholds:

        accepted = confidence >= threshold

        if accepted.sum() == 0:
            continue

        accepted_true = np.array(y_true)[accepted]

        accepted_predictions = np.array(
            predictions
        )[accepted]

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