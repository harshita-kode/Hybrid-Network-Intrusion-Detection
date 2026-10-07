import numpy as np
import pandas as pd
import shap

from src.schemas.results import (
    ExplanationResult,
    FeatureContribution,
)


class XGBoostExplainer:
    """
    Generate SHAP explanations for predictions made by
    the trained XGBoost intrusion-detection classifier.
    """

    def __init__(
        self,
        model,
        feature_names,
        label_classes,
        top_n=5
    ):
        self.model = model
        self.feature_names = list(feature_names)
        self.label_classes = list(label_classes)
        self.top_n = top_n

        if self.top_n <= 0:
            raise ValueError(
                "top_n must be greater than 0."
            )

        self.explainer = shap.TreeExplainer(
            self.model
        )

    def explain(self, sample):
        """
        Explain the XGBoost prediction for one network-flow sample.

        Returns:
            predicted_class
            confidence
            ExplanationResult
        """

        if not isinstance(sample, pd.DataFrame):
            raise TypeError(
                "Sample must be a pandas DataFrame."
            )

        if len(sample) != 1:
            raise ValueError(
                "Exactly one sample must be provided."
            )

        # --------------------------------------------------
        # VERIFY FEATURES
        # --------------------------------------------------

        missing_features = [
            feature
            for feature in self.feature_names
            if feature not in sample.columns
        ]

        if missing_features:
            raise ValueError(
                "Sample is missing required model features: "
                f"{missing_features}"
            )

        # Enforce the exact feature order used by XGBoost.
        sample = sample[
            self.feature_names
        ]

        # --------------------------------------------------
        # XGBOOST PREDICTION
        # --------------------------------------------------

        probabilities = self.model.predict_proba(
            sample
        )

        predicted_class_index = int(
            np.argmax(
                probabilities[0]
            )
        )

        confidence = float(
            probabilities[
                0,
                predicted_class_index
            ]
        )

        if (
            predicted_class_index
            >= len(self.label_classes)
        ):
            raise ValueError(
                "Predicted class index is outside "
                "the available label mapping."
            )

        predicted_class = str(
            self.label_classes[
                predicted_class_index
            ]
        )

        # --------------------------------------------------
        # SHAP EXPLANATION
        # --------------------------------------------------

        shap_explanation = self.explainer(
            sample
        )

        class_values = self._extract_class_values(
            shap_explanation.values,
            predicted_class_index,
            len(self.feature_names)
        )

        # --------------------------------------------------
        # FIND MOST INFLUENTIAL FEATURES
        # --------------------------------------------------

        ranked_indices = np.argsort(
            np.abs(class_values)
        )[::-1][:self.top_n]

        contributions = []

        for index in ranked_indices:

            contribution = FeatureContribution(
                feature=str(
                    sample.columns[index]
                ),
                feature_value=float(
                    sample.iloc[0, index]
                ),
                shap_value=float(
                    class_values[index]
                )
            )

            contributions.append(
                contribution
            )

        explanation = ExplanationResult(
            top_features=[
                contribution.feature
                for contribution in contributions
            ],
            contributions=contributions
        )

        return (
            predicted_class,
            confidence,
            explanation
        )

    @staticmethod
    def _extract_class_values(
        values,
        predicted_class_index,
        n_features
    ):
        """
        Extract SHAP values for the class predicted by XGBoost.
        """

        values = np.asarray(
            values
        )

        # Single-output format:
        # (samples, features)
        if values.ndim == 2:
            return values[0]

        # Multiclass formats.
        if values.ndim == 3:

            # Expected format:
            # (samples, features, classes)
            if values.shape[1] == n_features:
                return values[
                    0,
                    :,
                    predicted_class_index
                ]

            # Alternative format:
            # (samples, classes, features)
            if values.shape[2] == n_features:
                return values[
                    0,
                    predicted_class_index,
                    :
                ]

        raise ValueError(
            "Unsupported SHAP output shape: "
            f"{values.shape}"
        )