
"""
GC2 integration - Stage 1.

Real XGBoost classification, SHAP and security alerts.

Low-confidence traffic is marked for further analysis.
KNN/Autoencoder integration will be added in Stage 2/3.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.analysis.shap_data import load_shap_inputs
from src.analysis.explainability import XGBoostExplainer
from src.analysis.explanation_adapter import (
    adapt_shap_explanation
)
from src.pipeline.alert import generate_alert


class GC2Pipeline:

    def __init__(self, confidence_threshold=None):
        print("Initializing pipeline...")

        # Load the verified trained classifier.
        (
            self.model,
            _,
            _,
            self.feature_names,
            self.label_classes
        ) = load_shap_inputs()

        # Use the existing confidence configuration.
        from models.confidence import (
            DEFAULT_CONFIDENCE_THRESHOLD
        )

        self.confidence_threshold = (
            DEFAULT_CONFIDENCE_THRESHOLD
            if confidence_threshold is None
            else confidence_threshold
        )

        if not (
            isinstance(self.confidence_threshold, (int, float))
            and not isinstance(self.confidence_threshold, bool)
            and np.isfinite(self.confidence_threshold)
            and 0 <= self.confidence_threshold <= 1
        ):
            raise ValueError(
                "Invalid confidence threshold"
            )

        # Create SHAP explainer once.
        self.explainer = XGBoostExplainer(
            model=self.model,
            feature_names=self.feature_names,
            label_classes=self.label_classes,
            top_n=5
        )

        print("Stage 1 initialized.")

    def predict_one(self, flow, alert_id="GC2-001"):
        """Process one preprocessed network flow."""

        if isinstance(flow, dict):
            frame = pd.DataFrame([flow])
        elif isinstance(flow, pd.DataFrame):
            frame = flow.copy()
        else:
            raise TypeError(
                "Flow must be a dictionary or DataFrame"
            )

        if len(frame) != 1:
            raise ValueError(
                "Exactly one network flow is required"
            )

        missing = [
            name for name in self.feature_names
            if name not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"Missing XGBoost features: {missing}"
            )

        X = frame[self.feature_names].copy()

        for column in self.feature_names:
            if not pd.api.types.is_numeric_dtype(
                X[column]
            ):
                raise ValueError(
                    f"Non-numeric feature: {column}"
                )

        values = X.to_numpy(dtype=np.float64)

        if not np.isfinite(values).all():
            raise ValueError(
                "Network flow contains nonfinite values"
            )

        # -----------------------------------------------
        # REAL XGBOOST PREDICTION
        # -----------------------------------------------

        probabilities = self.model.predict_proba(X)[0]

        predicted_index = int(np.argmax(probabilities))

        predicted_class = str(
            self.label_classes[predicted_index]
        )

        confidence = float(
            probabilities[predicted_index]
        )

        # -----------------------------------------------
        # CONFIDENCE ROUTING
        # -----------------------------------------------

        high_confidence = (
            confidence >= self.confidence_threshold
        )

        if high_confidence:
            if predicted_class.strip().upper() == "BENIGN":
                status = "BENIGN"
            else:
                status = "KNOWN_ATTACK"
        else:
            # Stage 1 does not yet call KB/Autoencoder.
            status = "UNCERTAIN"

        # -----------------------------------------------
        # REAL SHAP EXPLANATION
        # -----------------------------------------------

        (
            explained_class,
            explained_confidence,
            shap_result
        ) = self.explainer.explain(X)

        if explained_class != predicted_class:
            raise ValueError(
                "SHAP prediction differs from XGBoost"
            )

        if not np.isclose(
            explained_confidence,
            confidence,
            rtol=1e-5,
            atol=1e-6
        ):
            raise ValueError(
                "SHAP confidence differs from XGBoost"
            )

        unified_shap = adapt_shap_explanation(
            shap_result
        )

        # -----------------------------------------------
        # DETECTION EVIDENCE
        # -----------------------------------------------

        evidence = {
            "classifier": "XGBoost",
            "confidence_threshold": float(
                self.confidence_threshold
            ),
            "confidence_route": (
                "HIGH_CONFIDENCE"
                if high_confidence
                else "LOW_CONFIDENCE"
            ),
            "further_analysis_required": (
                not high_confidence
            ),
            "integration_stage": 1,
        }

        # -----------------------------------------------
        # GENERATE ALERT
        # -----------------------------------------------

        # No asset ratings are inferred.
        # Risk will be connected in Stage 4.
        alert = generate_alert(
            alert_id=alert_id,
            detection_status=status,
            predicted_class=predicted_class,
            detection_confidence=confidence,
            evidence=evidence,
            explanations=[unified_shap],
            risk=None
        )

        return alert
