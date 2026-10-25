
"""Explain Version20 Autoencoder reconstruction errors."""

import numpy as np
import pandas as pd


def explain_autoencoder(flow, loaded_ae, top_n=5):
    """
    Explain one prepared network flow.

    loaded_ae:
        Tuple returned by kb_flow.load_version20(root).

    flow:
        One record with chronological behavioural
        features already calculated.

    Returns:
        Ensemble MSE, anomaly decision and
        per-feature reconstruction-error contributions.
    """

    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")

    common, preprocessing, models, threshold = loaded_ae

    if len(models) != 3:
        raise ValueError(
            "Version20 requires three Autoencoder models"
        )

    if isinstance(flow, dict):
        frame = pd.DataFrame([flow])
    elif isinstance(flow, pd.DataFrame):
        frame = flow.copy()
    else:
        raise TypeError(
            "flow must be a dictionary or DataFrame"
        )

    if len(frame) != 1:
        raise ValueError("Exactly one flow is required")

    missing_behavioural = (
        set(common.BEHAVIOURAL_FEATURES)
        - set(frame.columns)
    )

    if missing_behavioural:
        raise ValueError(
            "Missing chronological behavioural features: "
            f"{sorted(missing_behavioural)}"
        )

    # Apply the exact saved Version20 preprocessing.
    x = common.transform(frame, preprocessing)

    feature_names = list(preprocessing["columns"])

    if x.shape != (1, len(feature_names)):
        raise ValueError("Autoencoder input shape mismatch")

    # Average the squared reconstruction errors
    # for each feature across the three models.
    feature_errors = np.zeros(
        len(feature_names),
        dtype=np.float64
    )

    for model in models:
        reconstruction = np.asarray(
            model.predict(x, verbose=0)
        )

        if reconstruction.shape != x.shape:
            raise ValueError(
                "Autoencoder reconstruction shape mismatch"
            )

        squared_error = (
            x.astype(np.float64)
            - reconstruction.astype(np.float64)
        ) ** 2

        feature_errors += squared_error[0]

    feature_errors /= len(models)

    # Our feature-level explanation score.
    explanation_mse = float(feature_errors.mean())

    # Original scoring function from Member 3.
    original_mse = float(
        common.reconstruction_mse(models, x)[0]
    )

    if not np.isclose(
        explanation_mse,
        original_mse,
        rtol=1e-5,
        atol=1e-7
    ):
        raise ValueError(
            "Explanation MSE does not match "
            "the original Autoencoder score"
        )

    total_error = float(feature_errors.sum())

    ranked_indices = np.argsort(
        feature_errors
    )[::-1][:top_n]

    contributions = []

    for index in ranked_indices:
        contributions.append({
            "feature": feature_names[index],
            "transformed_value": float(x[0, index]),
            "squared_error": float(feature_errors[index]),
            "error_share": (
                float(feature_errors[index] / total_error)
                if total_error > 0
                else 0.0
            ),
        })

    return {
        "method": "autoencoder_reconstruction",
        "anomaly_mse": explanation_mse,
        "original_mse": original_mse,
        "threshold": float(threshold),
        "anomalous": bool(explanation_mse > threshold),
        "feature_count": len(feature_names),
        "ensemble_size": len(models),
        "top_features": contributions,
        "mse_consistency": True,
    }
