
"""
GC2 writable KB promotion verification.

Uses temporary databases only.
Does not change the production candidate database
or the original frozen KNN artifact.
"""

import hashlib
import tempfile
from pathlib import Path

import numpy as np

from src.pipeline.kb_candidates import KBCandidateStore
from src.pipeline.kb_promotion import WritableKB, FROZEN_KB


def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def expect_rejection(function):
    try:
        function()
    except (ValueError, Exception) as error:
        print("Expected rejection:", str(error))
        return

    raise AssertionError(
        "Operation should have been rejected"
    )


def main():
    original_hash = sha256(FROZEN_KB)

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)

        candidate_db = directory / "candidates.sqlite3"
        writable_db = directory / "writable.sqlite3"

        store = KBCandidateStore(candidate_db)

        kb = WritableKB(
            writable_path=writable_db,
            candidate_db=candidate_db,
        )

        # Use a synthetic vector deliberately far
        # from the frozen KB for promotion testing.
        synthetic = {
            column: 0.0
            for column in kb.columns
        }

        synthetic["Src Port"] = 65000
        synthetic["Dst Port"] = 65000
        synthetic["Flow Duration"] = 1e12

        evidence = {
            "detection_status": "POTENTIAL_NEW_UNKNOWN",
            "confidence_route": "LOW_CONFIDENCE",
            "kb_matched": False,
            "ae_anomalous": True,
        }

        candidate = store.add(
            synthetic,
            evidence
        )

        candidate_id = candidate["candidate_id"]

        # Pending candidates must not be promoted.
        expect_rejection(
            lambda: kb.promote(
                candidate_id,
                "Test Analyst",
                "Pending candidate test",
            )
        )

        # Approve in temporary test database.
        store.review(
            candidate_id,
            "APPROVED",
            "Test Analyst",
            "Synthetic promotion test",
        )

        promoted = kb.promote(
            candidate_id,
            "Test Analyst",
            "Controlled test promotion",
        )

        assert promoted["status"] == "PROMOTED"

        # Same candidate cannot be promoted twice.
        expect_rejection(
            lambda: kb.promote(
                candidate_id,
                "Test Analyst",
                "Duplicate promotion",
            )
        )

        # Promoted candidate must match itself.
        lookup = kb.lookup(synthetic)

        assert lookup["matched"] is True
        assert lookup["nearest_pattern_id"] == (
            promoted["pattern_id"]
        )

        assert np.isclose(
            lookup["nearest_distance"],
            0.0,
            atol=1e-6
        )

        assert len(kb.list_promoted()) == 1

        # Malformed candidate must be rejected.
        malformed = store.add(
            {"Src Port": 1234},
            evidence
        )

        store.review(
            malformed["candidate_id"],
            "APPROVED",
            "Test Analyst",
            "Malformed candidate test",
        )

        expect_rejection(
            lambda: kb.promote(
                malformed["candidate_id"],
                "Test Analyst",
                "Malformed candidate",
            )
        )

    # Frozen KB must remain byte-for-byte unchanged.
    assert sha256(FROZEN_KB) == original_hash

    print("\nApproved candidate promotion: PASSED")
    print("Pending candidate rejection: PASSED")
    print("Duplicate promotion rejection: PASSED")
    print("Writable KB lookup: PASSED")
    print("Malformed candidate rejection: PASSED")
    print("Frozen KB preservation: PASSED")
    print("\nALL CONTROLLED KB PROMOTION TESTS PASSED")


if __name__ == "__main__":
    main()
