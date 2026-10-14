
"""
GC2 positive shadow-KB integration verification.

Uses:
- Real known-test row 235964
- Temporary candidate database
- Temporary writable KB
- Real GC2 detection pipeline

Does not modify:
- Frozen KB
- Production candidate database
- Production writable KB

Shadow results must never override detection.
"""

import hashlib
import json
import tempfile

from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.integration import GC2Pipeline
from src.pipeline.ae_context import KnownTestContext
from src.pipeline.kb_candidates import KBCandidateStore
from src.pipeline.kb_promotion import WritableKB, FROZEN_KB
from src.pipeline.alert import alert_to_dict


TEST_ROW = 235964


def file_hash(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():

    print("\n========== SHADOW KB INTEGRATION TEST ==========")

    frozen_hash_before = file_hash(FROZEN_KB)

    # -------------------------------------------------
    # 1. Load real test flow and chronological context.
    # -------------------------------------------------

    print("Loading real known-test flow...")

    history = KnownTestContext()

    flow = history.data.iloc[[TEST_ROW]].copy()

    # -------------------------------------------------
    # 2. Initialize real GC2 pipeline.
    # -------------------------------------------------

    pipeline = GC2Pipeline()

    # -------------------------------------------------
    # 3. Run baseline detection.
    # -------------------------------------------------

    print("Running baseline detection...")

    # Use the same known-test context method that your
    # working run_gc2.py uses for this row.
    #
    # IMPORTANT:
    # The exact method name depends on your current
    # KnownTestContext implementation.
    #
    # Replace this call only if your class uses a
    # different method name.

    prepared_ae_flow, historical_flow_count = (
        history.prepare(TEST_ROW)
    )

    baseline = pipeline.predict_one(
        flow=flow,
        alert_id="GC2-SHADOW-BASELINE",
        prepared_ae_flow=prepared_ae_flow,
        historical_flow_count=historical_flow_count,
    )

    assert baseline.detection_status == (
        "POTENTIAL_NEW_UNKNOWN"
    )

    assert baseline.evidence["kb_matched"] is False
    assert baseline.evidence["ae_anomalous"] is True

    print("Baseline detection:", baseline.detection_status)

    # -------------------------------------------------
    # 4. Create temporary KB databases.
    # -------------------------------------------------

    with tempfile.TemporaryDirectory() as directory:

        directory = Path(directory)

        candidate_db = directory / "candidates.sqlite3"
        writable_db = directory / "writable.sqlite3"

        store = KBCandidateStore(candidate_db)

        kb = WritableKB(
            writable_path=writable_db,
            candidate_db=candidate_db,
        )

        # -------------------------------------------------
        # 5. Create candidate from real flow.
        # -------------------------------------------------

        flow_record = flow.iloc[0].to_dict()

        for column in (
            "Label",
            "Original_Label",
            "source_file",
            "Attempted Category",
            "is_attempted",
        ):
            flow_record.pop(column, None)

        evidence = {
            "detection_status": "POTENTIAL_NEW_UNKNOWN",
            "confidence_route": "LOW_CONFIDENCE",
            "kb_matched": False,
            "ae_anomalous": True,
            "ae_mse": baseline.evidence["ae_mse"],
        }

        candidate = store.add(
            flow=flow_record,
            evidence=evidence,
        )

        candidate_id = candidate["candidate_id"]

        # -------------------------------------------------
        # 6. Approve test candidate.
        # -------------------------------------------------

        store.review(
            candidate_id=candidate_id,
            decision="APPROVED",
            analyst="GC2 Test Analyst",
            note="Temporary shadow integration test",
        )

        # -------------------------------------------------
        # 7. Promote into temporary writable KB.
        # -------------------------------------------------

        promoted = kb.promote(
            candidate_id=candidate_id,
            promoted_by="GC2 Test Analyst",
            promotion_note="Positive shadow lookup verification",
        )

        print("Promoted pattern:", promoted["pattern_id"])

        # -------------------------------------------------
        # 8. Execute positive shadow lookup.
        # -------------------------------------------------

        shadow = kb.lookup(flow_record)

        assert shadow["matched"] is True

        assert shadow["nearest_pattern_id"] == (
            promoted["pattern_id"]
        )

        assert np.isclose(
            shadow["nearest_distance"],
            0.0,
            atol=1e-6,
        )

        assert shadow["mode"] == "SHADOW"

        print("Shadow matched:", shadow["matched"])
        print("Shadow distance:", shadow["nearest_distance"])

        # -------------------------------------------------
        # 9. Run real detection again.
        # -------------------------------------------------

        print("Running detection with shadow evidence...")

        alert = pipeline.predict_one(
            flow=flow,
            alert_id="GC2-SHADOW-POSITIVE",
            prepared_ae_flow=prepared_ae_flow,
            historical_flow_count=historical_flow_count,
        )

        # This mirrors the current runner's shadow
        # integration: lookup after detection.
        alert.evidence["writable_kb_shadow"] = kb.lookup(
            flow_record
        )

        # -------------------------------------------------
        # 10. Verify original detection is unchanged.
        # -------------------------------------------------

        assert alert.detection_status == (
            baseline.detection_status
        )

        assert alert.predicted_class == (
            baseline.predicted_class
        )

        assert np.isclose(
            alert.detection_confidence,
            baseline.detection_confidence,
        )

        assert alert.evidence["kb_matched"] == (
            baseline.evidence["kb_matched"]
        )

        assert alert.evidence["ae_anomalous"] == (
            baseline.evidence["ae_anomalous"]
        )

        assert np.isclose(
            alert.evidence["ae_mse"],
            baseline.evidence["ae_mse"],
            rtol=1e-5,
            atol=1e-7,
        )

        assert alert.evidence[
            "writable_kb_shadow"
        ]["matched"] is True

        # -------------------------------------------------
        # 11. Verify JSON serialization.
        # -------------------------------------------------

        serialized = alert_to_dict(alert)

        json.dumps(
            serialized,
            allow_nan=False,
        )

        assert serialized[
            "evidence"
        ]["writable_kb_shadow"]["matched"] is True

        print("Detection unchanged: PASSED")
        print("JSON serialization: PASSED")

    # -------------------------------------------------
    # 12. Verify frozen KB preservation.
    # -------------------------------------------------

    frozen_hash_after = file_hash(FROZEN_KB)

    assert frozen_hash_before == frozen_hash_after

    print("Frozen KB unchanged: PASSED")

    print("\n========== FINAL RESULTS ==========")
    print("Original status:", baseline.detection_status)
    print("Shadow match:", shadow["matched"])
    print("Shadow pattern:", shadow["nearest_pattern_id"])
    print("Shadow distance:", shadow["nearest_distance"])
    print("Original KB updated: False")

    print("\nALL POSITIVE SHADOW KB TESTS PASSED")


if __name__ == "__main__":
    main()
