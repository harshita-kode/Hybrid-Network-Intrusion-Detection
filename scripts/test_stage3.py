
"""
GC2 Stage 3 integration test.

Real XGBoost -> KNN miss -> Version20 Autoencoder
-> unified explanations -> JSON security alert.

No Knowledge Base updates are performed.
"""

import sys
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.integration import GC2Pipeline
from src.pipeline.ae_context import KnownTestContext
from src.pipeline.ae_integration import AutoencoderIntegration

from src.pipeline.alert import (
    generate_alert,
    format_alert,
    alert_to_dict,
)


TEST_ROW = 235964

OUTPUT_PATH = (
    ROOT
    / "outputs"
    / "alerts"
    / "GC2-STAGE3-TEST.json"
)


def main():

    print("\n========== GC2 STAGE 3 TEST ==========")

    # --------------------------------------------------
    # LOAD CHRONOLOGICAL CONTEXT
    # --------------------------------------------------

    history = KnownTestContext()

    original_flow = history.data.iloc[[TEST_ROW]].copy()

    prepared_flow, prior_count = history.prepare(
        TEST_ROW
    )

    print("Original test row:", TEST_ROW)
    print("Historical flows:", prior_count)

    # --------------------------------------------------
    # RUN REAL STAGE 2 PIPELINE
    # --------------------------------------------------

    pipeline = GC2Pipeline()

    stage2_alert = pipeline.predict_one(
        original_flow,
        alert_id="GC2-STAGE3-TEST"
    )

    evidence = dict(stage2_alert.evidence)

    if evidence.get("confidence_route") != "LOW_CONFIDENCE":
        raise ValueError(
            "Selected flow did not enter low-confidence route"
        )

    if evidence.get("kb_called") is not True:
        raise ValueError(
            "KNN was not called"
        )

    if evidence.get("kb_matched") is not False:
        raise ValueError(
            "Selected flow is not a KB miss"
        )

    print("\nStage 2 routing: PASSED")

    # --------------------------------------------------
    # RUN REAL AUTOENCODER
    # --------------------------------------------------

    ae = AutoencoderIntegration()

    ae_result, ae_explanation = ae.analyze(
        prepared_flow
    )

    # Verify independently against the original scorer.
    from notebooks.kb.kb_flow import score_precomputed_flow

    original_mse, original_threshold = (
        score_precomputed_flow(
            prepared_flow,
            ae.loaded_ae
        )
    )

    np.testing.assert_allclose(
        ae_result["anomaly_mse"],
        original_mse,
        rtol=1e-5,
        atol=1e-7
    )

    np.testing.assert_allclose(
        ae_result["threshold"],
        original_threshold,
        rtol=1e-5,
        atol=1e-7
    )

    anomalous = bool(ae_result["anomalous"])

    # --------------------------------------------------
    # FINAL DETECTION STATUS
    # --------------------------------------------------

    if anomalous:
        final_status = "POTENTIAL_NEW_UNKNOWN"
        further_analysis = True
    else:
        final_status = "UNCERTAIN"
        further_analysis = True

    evidence.update({
        "integration_stage": 3,
        "ae_called": True,
        "ae_anomalous": anomalous,
        "ae_mse": float(ae_result["anomaly_mse"]),
        "ae_threshold": float(ae_result["threshold"]),
        "historical_flow_count": prior_count,
        "history_source": "known_test.parquet",
        "kb_updated": False,
        "further_analysis_required": further_analysis,
        "review_reason": (
            "KB miss followed by Autoencoder anomaly"
            if anomalous
            else "KB miss without Autoencoder anomaly"
        ),
    })

    # Preserve existing SHAP and KNN explanations.
    explanations = list(stage2_alert.explanations)

    # Add the real Autoencoder explanation.
    explanations.append(ae_explanation)

    # --------------------------------------------------
    # GENERATE FINAL ALERT
    # --------------------------------------------------

    final_alert = generate_alert(
        alert_id="GC2-STAGE3-TEST",
        detection_status=final_status,
        predicted_class=stage2_alert.predicted_class,
        detection_confidence=stage2_alert.detection_confidence,
        evidence=evidence,
        explanations=explanations,
        risk=None,
    )

    assert len(final_alert.explanations) == 3

    assert [
        item.method
        for item in final_alert.explanations
    ] == [
        "shap",
        "knn_feature_distance",
        "autoencoder_reconstruction",
    ]

    assert final_alert.evidence["kb_updated"] is False

    print("\n" + format_alert(final_alert))

    # --------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            alert_to_dict(final_alert),
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    print("\nSaved JSON:", OUTPUT_PATH)

    print("\nSTAGE 3 INTEGRATION TEST: PASSED")


if __name__ == "__main__":
    main()
