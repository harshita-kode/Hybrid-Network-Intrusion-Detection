
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.shap_data import load_shap_inputs
from src.analysis.explainability import XGBoostExplainer


def main():
    model, X, y, features, classes = load_shap_inputs()

    sample = X.sample(
        n=min(20, len(X)),
        random_state=42
    )

    explainer = XGBoostExplainer(
        model=model,
        feature_names=features,
        label_classes=classes,
        top_n=5
    )

    explanation = explainer.explainer(sample)

    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)

    print("SHAP values shape:", values.shape)
    print("SHAP baseline shape:", base.shape)

    if values.shape != (
        len(sample),
        len(features),
        len(classes)
    ):
        raise ValueError(
            f"Unexpected multiclass SHAP shape: {values.shape}"
        )

    # Calculate raw XGBoost margins, not probabilities.
    margins = np.asarray(
        model.get_booster().predict(
            xgb.DMatrix(
                sample,
                feature_names=features
            ),
            output_margin=True
        )
    )

    reconstructed = (
        base + values.sum(axis=1)
    )

    if reconstructed.shape != margins.shape:
        raise ValueError("SHAP/margin shape mismatch")

    errors = np.abs(reconstructed - margins)

    print("Maximum absolute error:", errors.max())
    print("Mean absolute error:", errors.mean())

    np.testing.assert_allclose(
        reconstructed,
        margins,
        rtol=1e-4,
        atol=1e-4
    )

    print("SHAP ADDITIVITY: PASSED")


if __name__ == "__main__":
    main()
