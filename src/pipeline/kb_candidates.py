
"""
GC2 candidate unknown-pattern store.

Stores potential unknown patterns for analyst review.

Does NOT update the frozen KNN Knowledge Base.
"""

import hashlib
import json
import sqlite3

from datetime import datetime, timezone
from pathlib import Path

from contextlib import contextmanager


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DB = (
    ROOT / "outputs" / "kb_candidates" / "candidates.sqlite3"
)

ALLOWED_DECISIONS = {"APPROVED", "REJECTED"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_text(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


class KBCandidateStore:

    def __init__(self, db_path=DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.initialize()

    
    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row

        try:
            with connection:
                yield connection
        finally:
            connection.close()


    def initialize(self):
        with self.connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    observation_count INTEGER NOT NULL,
                    flow_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    analyst TEXT,
                    review_note TEXT,
                    reviewed_at TEXT
                )
            """)

    def add(self, flow, evidence):
        """
        Store a potential unknown pattern.

        Identical canonical flow content receives
        the same fingerprint.

        Existing reviewed decisions are preserved.
        """

        flow_payload = json_text(flow)
        evidence_payload = json_text(evidence)

        fingerprint = hashlib.sha256(
            flow_payload.encode("utf-8")
        ).hexdigest()

        candidate_id = "CAND-" + fingerprint[:16]

        now = utc_now()

        with self.connect() as connection:
            existing = connection.execute(
                """
                SELECT candidate_id, status
                FROM candidates
                WHERE fingerprint = ?
                """,
                (fingerprint,),
            ).fetchone()

            if existing is not None:
                connection.execute(
                    """
                    UPDATE candidates
                    SET last_seen = ?,
                        observation_count = observation_count + 1
                    WHERE fingerprint = ?
                    """,
                    (now, fingerprint),
                )

                return {
                    "candidate_id": existing["candidate_id"],
                    "created": False,
                    "status": existing["status"],
                }

            connection.execute(
                """
                INSERT INTO candidates (
                    candidate_id,
                    fingerprint,
                    status,
                    first_seen,
                    last_seen,
                    observation_count,
                    flow_json,
                    evidence_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    fingerprint,
                    "PENDING_REVIEW",
                    now,
                    now,
                    1,
                    flow_payload,
                    evidence_payload,
                ),
            )

        return {
            "candidate_id": candidate_id,
            "created": True,
            "status": "PENDING_REVIEW",
        }

    def review(self, candidate_id, decision, analyst, note):
        decision = str(decision).upper()

        if decision not in ALLOWED_DECISIONS:
            raise ValueError(
                "Decision must be APPROVED or REJECTED"
            )

        if not analyst or not analyst.strip():
            raise ValueError("Analyst is required")

        if not note or not note.strip():
            raise ValueError("Review note is required")

        with self.connect() as connection:
            existing = connection.execute(
                """
                SELECT status
                FROM candidates
                WHERE candidate_id = ?
                """,
                (candidate_id,),
            ).fetchone()

            if existing is None:
                raise ValueError("Unknown candidate ID")

            if existing["status"] != "PENDING_REVIEW":
                raise ValueError(
                    "Candidate has already been reviewed"
                )

            connection.execute(
                """
                UPDATE candidates
                SET status = ?,
                    analyst = ?,
                    review_note = ?,
                    reviewed_at = ?
                WHERE candidate_id = ?
                """,
                (
                    decision,
                    analyst.strip(),
                    note.strip(),
                    utc_now(),
                    candidate_id,
                ),
            )

        return self.get(candidate_id)

    def get(self, candidate_id):
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM candidates
                WHERE candidate_id = ?
                """,
                (candidate_id,),
            ).fetchone()

        if row is None:
            return None

        result = dict(row)

        result["flow"] = json.loads(
            result.pop("flow_json")
        )

        result["evidence"] = json.loads(
            result.pop("evidence_json")
        )

        return result

    def list_candidates(self, status=None):
        with self.connect() as connection:
            if status is None:
                rows = connection.execute(
                    """
                    SELECT candidate_id, status,
                           first_seen, last_seen,
                           observation_count
                    FROM candidates
                    ORDER BY first_seen DESC
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT candidate_id, status,
                           first_seen, last_seen,
                           observation_count
                    FROM candidates
                    WHERE status = ?
                    ORDER BY first_seen DESC
                    """,
                    (status,),
                ).fetchall()

        return [dict(row) for row in rows]
