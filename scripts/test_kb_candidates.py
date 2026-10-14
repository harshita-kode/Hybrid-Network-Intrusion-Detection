
"""Verify GC2 candidate storage and review."""

import tempfile
from pathlib import Path

from src.pipeline.kb_candidates import KBCandidateStore


def main():
    with tempfile.TemporaryDirectory() as directory:
        db_path = Path(directory) / "test_candidates.sqlite3"

        store = KBCandidateStore(db_path)

        flow = {
            "Src Port": 46592,
            "Dst Port": 22,
            "Flow Duration": 12039789,
        }

        evidence = {
            "detection_status": "POTENTIAL_NEW_UNKNOWN",
            "confidence_route": "LOW_CONFIDENCE",
            "kb_matched": False,
            "ae_anomalous": True,
            "ae_mse": 0.5278878,
        }

        first = store.add(flow, evidence)

        assert first["created"] is True
        assert first["status"] == "PENDING_REVIEW"

        second = store.add(flow, evidence)

        assert second["created"] is False
        assert second["candidate_id"] == first["candidate_id"]

        stored = store.get(first["candidate_id"])

        assert stored["observation_count"] == 2
        assert stored["status"] == "PENDING_REVIEW"

        pending = store.list_candidates(
            status="PENDING_REVIEW"
        )

        assert len(pending) == 1

        reviewed = store.review(
            candidate_id=first["candidate_id"],
            decision="APPROVED",
            analyst="Test Analyst",
            note="Controlled review test",
        )

        assert reviewed["status"] == "APPROVED"

        try:
            store.review(
                candidate_id=first["candidate_id"],
                decision="REJECTED",
                analyst="Another Analyst",
                note="Second decision",
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Repeated review was incorrectly accepted"
            )

        # Duplicate observations must not reset review.
        third = store.add(flow, evidence)

        assert third["status"] == "APPROVED"

        stored = store.get(first["candidate_id"])

        assert stored["observation_count"] == 3

        print("Candidate insertion: PASSED")
        print("Duplicate detection: PASSED")
        print("Observation counting: PASSED")
        print("Analyst approval: PASSED")
        print("Repeated review rejection: PASSED")
        print("Reviewed status preservation: PASSED")
        print()
        print("ALL KB CANDIDATE TESTS PASSED")


if __name__ == "__main__":
    main()
