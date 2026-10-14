
"""
GC2 integration - Stage 2.

Real XGBoost + SHAP + KNN/KB + Security Alerts.

High confidence:
    XGBoost classification + SHAP

Low confidence:
    KNN lookup + SHAP + KB explanation

KB misses remain UNCERTAIN until Autoencoder integration.
KB matches are reported as similarity evidence, not
automatically as confirmed attacks.

The frozen Knowledge Base is never modified.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from src.analysis.risk_lookup import RiskLookup

from src.analysis.historical_similarity import HistoricalSimilarity

from src.pipeline.ae_integration import AutoencoderIntegration


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from src.analysis.shap_data import load_shap_inputs

from src.analysis.explainability import XGBoostExplainer

from src.analysis.kb_explanation import explain_kb_match

from src.analysis.explanation_adapter import (
    adapt_shap_explanation,
    adapt_kb_explanation,
)

from src.pipeline.alert import generate_alert

from models.confidence import DEFAULT_CONFIDENCE_THRESHOLD


KB_MODEL_PATH = (
    ROOT / "saved-models"
    / "knn_kb"
    / "initial_kb.joblib"
)

KB_REFERENCE_PATH = (
    ROOT / "data"
    / "processed"
    / "knn_kb"
    / "knowledge_base.csv"
)


class GC2Pipeline:

    def __init__(self, confidence_threshold=None):

        print("Initializing detection pipeline...")

        (
            self.model,
            _,
            _,
            self.feature_names,
            self.label_classes
        ) = load_shap_inputs()

        self.ae_integration = None
        self.risk_lookup = RiskLookup()

        print("Loading historical attack-profile comparator...")

        self.historical_comparator = HistoricalSimilarity()

        self.confidence_threshold = (
            DEFAULT_CONFIDENCE_THRESHOLD
            if confidence_threshold is None
            else confidence_threshold
        )

        threshold = self.confidence_threshold

        if (
            isinstance(threshold, bool)
            or not isinstance(
                threshold, (int, float, np.integer, np.floating)
            )
            or not np.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError(
                "Confidence threshold must be between 0 and 1"
            )

        self.confidence_threshold = float(threshold)

        self.explainer = XGBoostExplainer(
            model=self.model,
            feature_names=self.feature_names,
            label_classes=self.label_classes,
            top_n=5
        )


        # --------------------------------------------------
        # LOAD FROZEN KNN KNOWLEDGE BASE
        # --------------------------------------------------

        print("Loading frozen KNN Knowledge Base...")

        if not KB_MODEL_PATH.is_file():
            raise FileNotFoundError(KB_MODEL_PATH)

        if not KB_REFERENCE_PATH.is_file():
            raise FileNotFoundError(KB_REFERENCE_PATH)

        self.kb_artifact = joblib.load(KB_MODEL_PATH)

        self.kb_reference = pd.read_csv(
            KB_REFERENCE_PATH,
            low_memory=False
        )

        if len(self.kb_reference) != self.kb_artifact["count"]:
            raise ValueError(
                "Knowledge Base reference/index count mismatch"
            )

        if "pattern_id" not in self.kb_reference.columns:
            raise ValueError(
                "KB reference must contain pattern_id"
            )

        kb_features = list(self.kb_artifact["columns"])

        if len(set(kb_features)) != len(kb_features):
            raise ValueError(
                "Duplicate KB feature names"
            )

        missing_reference_features = [
            name
            for name in kb_features
            if name not in self.kb_reference.columns
        ]

        if missing_reference_features:
            raise ValueError(
                "KB reference missing features: "
                f"{missing_reference_features}"
            )

        print("KB records:", len(self.kb_reference))
        print("KB features:", len(kb_features))
        print("KB cutoff:", self.kb_artifact["cutoff"])

        print("Detection pipeline initialized.")

    def predict_one(
        self,
        flow,
        alert_id="GC2-001",
        prepared_ae_flow=None,
        historical_flow_count=None,
        asset_id=None,
        acceptable_risk=None,
    ):

        # --------------------------------------------------
        # PREPARE INCOMING FLOW
        # --------------------------------------------------

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
            name
            for name in self.feature_names
            if name not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"Missing XGBoost features: {missing}"
            )

        X = frame[self.feature_names].copy()

        for name in self.feature_names:
            if not pd.api.types.is_numeric_dtype(X[name]):
                raise ValueError(
                    f"Non-numeric XGBoost feature: {name}"
                )

        values = X.to_numpy(dtype=np.float64)

        if not np.isfinite(values).all():
            raise ValueError(
                "Incoming flow contains nonfinite XGBoost values"
            )

        # --------------------------------------------------
        # XGBOOST CLASSIFICATION
        # --------------------------------------------------

        probabilities = self.model.predict_proba(X)[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = str(
            self.label_classes[predicted_index]
        )

        confidence = float(
            probabilities[predicted_index]
        )

        high_confidence = (
            confidence >= self.confidence_threshold
        )

        # --------------------------------------------------
        # SHAP EXPLANATION
        # --------------------------------------------------

        (
            explained_class,
            explained_confidence,
            shap_result
        ) = self.explainer.explain(X)

        if explained_class != predicted_class:
            raise ValueError(
                "SHAP and XGBoost predictions differ"
            )

        if not np.isclose(
            explained_confidence,
            confidence,
            rtol=1e-5,
            atol=1e-6
        ):
            raise ValueError(
                "SHAP and XGBoost confidence differ"
            )

        explanations = [
            adapt_shap_explanation(shap_result)
        ]

        # --------------------------------------------------
        # COMMON EVIDENCE
        # --------------------------------------------------

        evidence = {
            "classifier": "XGBoost",
            "confidence_threshold": self.confidence_threshold,
            "confidence_route": (
                "HIGH_CONFIDENCE"
                if high_confidence
                else "LOW_CONFIDENCE"
            ),
            "further_analysis_required": not high_confidence,
            "integration_stage": 2,
        }

        # --------------------------------------------------
        # HIGH CONFIDENCE
        # --------------------------------------------------

        if high_confidence:

            if predicted_class.upper() == "BENIGN":
                status = "BENIGN"
            else:
                status = "KNOWN_ATTACK"

            evidence["kb_called"] = False
            evidence["ae_called"] = False

        # --------------------------------------------------
        # LOW CONFIDENCE -> KNN
        # --------------------------------------------------

        else:

            print(
                "\nLow-confidence prediction."
                " Checking Knowledge Base..."
            )

            kb_features = list(
                self.kb_artifact["columns"]
            )

            missing_kb = [
                name
                for name in kb_features
                if name not in frame.columns
            ]

            if missing_kb:
                raise ValueError(
                    "Incoming flow is missing KB features: "
                    f"{missing_kb}"
                )

            # Pass the original flow, not XGBoost's
            # selected feature matrix.
            kb_result = explain_kb_match(
                flow=frame,
                artifact=self.kb_artifact,
                reference=self.kb_reference,
                top_n=5
            )

            explanations.append(
                adapt_kb_explanation(kb_result)
            )

            kb_matched = bool(
                kb_result["kb_matched"]
            )

            evidence.update({
                "kb_called": True,
                "kb_matched": kb_matched,
                "nearest_pattern_id": kb_result[
                    "nearest_pattern_id"
                ],
                "nearest_distance": float(
                    kb_result["nearest_distance"]
                ),
                "kb_cutoff": float(
                    kb_result["match_cutoff"]
                ),
                "ae_called": False,
                "kb_updated": False,
            })

            if kb_matched:
                # Similarity alone does not prove that
                # the stored record is a confirmed attack.
                status = "UNCERTAIN"

                evidence["kb_route"] = "KB_MATCH"
                evidence["further_analysis_required"] = True
                evidence["review_reason"] = (
                    "Stored KB pattern matched; "
                    "pattern provenance is not verified"
                )


                        
            else:
                # KB MISS -> AUTOENCODER

                status = "UNCERTAIN"
                evidence["kb_route"] = "AUTOENCODER"
                evidence["further_analysis_required"] = True

                if prepared_ae_flow is None:
                    # Do not invent historical behavioural features.
                    evidence["ae_called"] = False
                    evidence["review_reason"] = (
                        "KB miss; chronological context unavailable"
                    )

                else:
                    # Load the ensemble only when it is needed.
                    if self.ae_integration is None:
                        self.ae_integration = AutoencoderIntegration()

                    ae_result, ae_explanation = (
                        self.ae_integration.analyze(
                            prepared_ae_flow
                        )
                    )

                    explanations.append(ae_explanation)

                    anomalous = bool(ae_result["anomalous"])

                    evidence.update({
                        "ae_called": True,
                        "ae_anomalous": anomalous,
                        "ae_mse": float(ae_result["anomaly_mse"]),
                        "ae_threshold": float(ae_result["threshold"]),
                        "historical_flow_count": historical_flow_count,
                        "kb_updated": False,
                        "integration_stage": 3,
                    })

                    if anomalous:
                        status = "POTENTIAL_NEW_UNKNOWN"
                        evidence["review_reason"] = (
                            "KB miss followed by Autoencoder anomaly"
                        )
                    else:
                        status = "UNCERTAIN"
                        evidence["review_reason"] = (
                            "KB miss without Autoencoder anomaly"
                        )


        # --------------------------------------------------
        # SECURITY ALERT
        # --------------------------------------------------

        
        # --------------------------------------------------
        # ASSET-BASED RISK ASSESSMENT
        # --------------------------------------------------

        destination_ip = None
        destination_port = None

        if "Dst IP" in frame.columns:
            value = frame.iloc[0]["Dst IP"]

            if pd.notna(value):
                destination_ip = str(value)

        if "Dst Port" in frame.columns:
            value = frame.iloc[0]["Dst Port"]

            if pd.notna(value):
                destination_port = int(value)

        risk_result = self.risk_lookup.assess(
            detection_status=status,
            predicted_class=predicted_class,
            detection_confidence=confidence,
            asset_id=asset_id,
            destination_ip=destination_ip,
            destination_port=destination_port,
            acceptable_risk=acceptable_risk,
        )

        evidence["risk_config_illustrative"] = (
            self.risk_lookup.illustrative
        )

        evidence["risk_calculated"] = (
            risk_result is not None
        )

        if risk_result is None:
            evidence["risk_note"] = (
                "No accepted known attack or no matching "
                "asset/threat rating configuration"
            )
        else:
            evidence["risk_note"] = (
                "Illustrative risk assessment for testing"
                if self.risk_lookup.illustrative
                else "Configured asset-based risk assessment"
            )

        
        # ==================================================
        # HISTORICAL ATTACK-PROFILE COMPARISON
        # ==================================================

        historical_result = self.historical_comparator.compare(
            frame
        )

        evidence["historical_comparison"] = historical_result

        evidence["historical_comparison_role"] = (
            "SUPPORTING_EVIDENCE_ONLY"
        )



        return generate_alert(
            alert_id=alert_id,
            detection_status=status,
            predicted_class=predicted_class,
            detection_confidence=confidence,
            evidence=evidence,
            explanations=explanations,
            risk=risk_result
        )
