import numpy as np
import pandas as pd
import shap

from src.schemas.results import (
    ExplanationResult,
    FeatureContribution,
)


class XGBoostExplainer:
    """
    Generate local and global SHAP explanations for predictions
    made by the trained XGBoost intrusion-detection classifier.
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

        # TreeExplainer is appropriate for the XGBoost model.
        self.explainer = shap.TreeExplainer(
            self.model
        )

    # ========================================================
    # INDIVIDUAL / LOCAL EXPLANATION
    # ========================================================

    def explain(self, sample):
        """
        Explain the XGBoost prediction for one network-flow sample.

        Parameters
        ----------
        sample : pandas.DataFrame
            Exactly one processed network-flow sample.

        Returns
        -------
        predicted_class : str
            Predicted traffic/attack class.

        confidence : float
            XGBoost confidence for the predicted class.

        explanation : ExplanationResult
            Top SHAP feature contributions for the prediction.
        """

        if not isinstance(sample, pd.DataFrame):
            raise TypeError(
                "Sample must be a pandas DataFrame."
            )

        if len(sample) != 1:
            raise ValueError(
                "Exactly one sample must be provided."
            )

        # ----------------------------------------------------
        # VERIFY FEATURES
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # XGBOOST PREDICTION
        # ----------------------------------------------------

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

        if predicted_class_index >= len(
            self.label_classes
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

        # ----------------------------------------------------
        # SHAP EXPLANATION
        # ----------------------------------------------------

        shap_explanation = self.explainer(
            sample
        )

        class_values = self._extract_class_values(
            shap_explanation.values,
            predicted_class_index,
            len(self.feature_names)
        )

        # ----------------------------------------------------
        # FIND MOST INFLUENTIAL FEATURES
        # ----------------------------------------------------

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

    # ========================================================
    # GLOBAL EXPLANATION
    # ========================================================

    def explain_global(
        self,
        data,
        top_n=15
    ):
        """
        Calculate global SHAP feature importance across
        multiple network-flow samples.

        Global importance is calculated using the mean
        absolute SHAP value for each feature.

        Parameters
        ----------
        data : pandas.DataFrame
            Multiple processed network-flow samples.

        top_n : int
            Number of globally important features to return.

        Returns
        -------
        pandas.DataFrame
            Feature names and global SHAP importance,
            ordered from most to least influential.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Data must be a pandas DataFrame."
            )

        if len(data) == 0:
            raise ValueError(
                "Data cannot be empty."
            )

        if top_n <= 0:
            raise ValueError(
                "top_n must be greater than 0."
            )

        missing_features = [
            feature
            for feature in self.feature_names
            if feature not in data.columns
        ]

        if missing_features:
            raise ValueError(
                "Data is missing required model features: "
                f"{missing_features}"
            )

        # Enforce exact model feature order.
        data = data[
            self.feature_names
        ]

        # ----------------------------------------------------
        # CALCULATE SHAP VALUES
        # ----------------------------------------------------

        shap_explanation = self.explainer(
            data
        )

        values = np.asarray(
            shap_explanation.values
        )

        # ----------------------------------------------------
        # GLOBAL FEATURE IMPORTANCE
        # ----------------------------------------------------

        if values.ndim == 2:

            # Shape:
            # (samples, features)

            mean_absolute_shap = np.mean(
                np.abs(values),
                axis=0
            )

        elif values.ndim == 3:

            if values.shape[1] == len(
                self.feature_names
            ):

                # Shape:
                # (samples, features, classes)

                mean_absolute_shap = np.mean(
                    np.abs(values),
                    axis=(0, 2)
                )

            elif values.shape[2] == len(
                self.feature_names
            ):

                # Alternative shape:
                # (samples, classes, features)

                mean_absolute_shap = np.mean(
                    np.abs(values),
                    axis=(0, 1)
                )

            else:

                raise ValueError(
                    "Could not identify feature dimension "
                    f"in SHAP shape {values.shape}."
                )

        else:

            raise ValueError(
                "Unsupported SHAP output shape: "
                f"{values.shape}"
            )

        # ----------------------------------------------------
        # CREATE GLOBAL IMPORTANCE TABLE
        # ----------------------------------------------------

        importance = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_absolute_shap
        })

        importance = importance.sort_values(
            by="mean_abs_shap",
            ascending=False
        ).reset_index(
            drop=True
        )

        return importance.head(
            top_n
        )

    # ========================================================
    # SHAP OUTPUT HANDLING
    # ========================================================

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

            # Common format:
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