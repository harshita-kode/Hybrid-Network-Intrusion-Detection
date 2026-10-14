
"""
GC2 controlled KB promotion.

Promotes explicitly approved candidates into a separate
writable Knowledge Base.

The original frozen KB is never modified.

The writable KB initially operates in shadow mode:
its matches are supporting evidence and do not change
the main GC2 detection decision.
"""

import hashlib
import json
import sqlite3

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pipeline.kb_candidates import KBCandidateStore


ROOT = Path(__file__).resolve().parents[2]

FROZEN_KB = (
    ROOT / "saved-models" / "knn_kb" / "initial_kb.joblib"
)

WRITABLE_DB = (
    ROOT / "outputs" / "kb_candidates" / "writable_kb.sqlite3"
)

# Diagnostic numerical tolerance, not a similarity cutoff.
NUMERIC_TOLERANCE = 1e-6


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def database_connection(path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row

    try:
        with connection:
            yield connection
    finally:
        connection.close()


class WritableKB:

    def __init__(
        self,
        frozen_path=FROZEN_KB,
        writable_path=WRITABLE_DB,
        candidate_db=None,
    ):
        self.frozen_path = Path(frozen_path)
        self.writable_path = Path(writable_path)

        self.writable_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.frozen = joblib.load(self.frozen_path)

        self.columns = list(self.frozen["columns"])
        self.imputer = self.frozen["imputer"]
        self.scaler = self.frozen["scaler"]
        self.cutoff = float(self.frozen["cutoff"])

        self.frozen_vectors = np.asarray(
            self.frozen["vectors"],
            dtype=np.float64
        )

        self.frozen_index = self.frozen["index"]

        if len(self.columns) != 73:
            raise ValueError(
                "Unexpected frozen KB feature count"
            )

        if self.frozen_vectors.shape[1] != len(self.columns):
            raise ValueError(
                "Frozen KB feature mismatch"
            )

        self.frozen_hash = self._file_hash(
            self.frozen_path
        )

        self.candidates = (
            KBCandidateStore(candidate_db)
            if candidate_db is not None
            else KBCandidateStore()
        )

        self.initialize()

    @staticmethod
    def _file_hash(path):
        digest = hashlib.sha256()

        with Path(path).open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b""
            ):
                digest.update(chunk)

        return digest.hexdigest()

    def connect(self):
        return database_connection(
            self.writable_path
        )

    def initialize(self):
        with self.connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS promoted_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL UNIQUE,
                    vector_json TEXT NOT NULL,
                    vector_hash TEXT NOT NULL UNIQUE,
                    promoted_at TEXT NOT NULL,
                    promoted_by TEXT NOT NULL,
                    promotion_note TEXT NOT NULL,
                    frozen_hash TEXT NOT NULL
                )
            """)

    def transform_flow(self, flow):
        """
        Apply the exact saved 73-feature preprocessing.
        Never refit the imputer or scaler.
        """

        missing = [
            column
            for column in self.columns
            if column not in flow
        ]

        if missing:
            raise ValueError(
                f"Candidate missing KB features: {missing}"
            )

        values = pd.DataFrame(
            [[flow[column] for column in self.columns]],
            columns=self.columns
        )

        values = values.apply(
            pd.to_numeric,
            errors="raise"
        )

        raw = values.to_numpy(dtype=np.float64)

        if not np.isfinite(raw).all():
            raise ValueError(
                "Candidate contains nonfinite KB features"
            )

        imputed = self.imputer.transform(values)
        scaled = self.scaler.transform(imputed)

        vector = np.asarray(
            scaled[0],
            dtype=np.float64
        )

        if vector.shape != (len(self.columns),):
            raise ValueError(
                "Transformed KB vector has invalid shape"
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                "Transformed KB vector is nonfinite"
            )

        return vector

    @staticmethod
    def vector_hash(vector):
        normalized = np.round(
            np.asarray(vector, dtype=np.float64),
            decimals=10
        )

        return hashlib.sha256(
            normalized.tobytes()
        ).hexdigest()

    def _load_patterns(self):
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT pattern_id, candidate_id,
                       vector_json, frozen_hash
                FROM promoted_patterns
                ORDER BY promoted_at, pattern_id
                """
            ).fetchall()

        patterns = []

        for row in rows:
            if row["frozen_hash"] != self.frozen_hash:
                raise ValueError(
                    "Writable KB references a different "
                    "frozen KB artifact"
                )

            vector = np.asarray(
                json.loads(row["vector_json"]),
                dtype=np.float64
            )

            if (
                vector.shape != (len(self.columns),)
                or not np.isfinite(vector).all()
            ):
                raise ValueError(
                    "Stored writable KB vector is invalid"
                )

            patterns.append({
                "pattern_id": row["pattern_id"],
                "candidate_id": row["candidate_id"],
                "vector": vector,
            })

        return patterns

    def promote(
        self,
        candidate_id,
        promoted_by,
        promotion_note,
    ):
        """
        Explicitly promote an approved candidate.

        Reject:
        - Unapproved candidates.
        - Missing or invalid features.
        - Existing candidate promotions.
        - Exact transformed duplicates.
        - Candidates already matching the frozen KB.
        """

        if not promoted_by or not promoted_by.strip():
            raise ValueError(
                "Promoter identity is required"
            )

        if not promotion_note or not promotion_note.strip():
            raise ValueError(
                "Promotion note is required"
            )

        candidate = self.candidates.get(
            candidate_id
        )

        if candidate is None:
            raise ValueError(
                "Candidate not found"
            )

        if candidate["status"] != "APPROVED":
            raise ValueError(
                "Only approved candidates can be promoted"
            )

        evidence = candidate["evidence"]

        if not (
            evidence.get("detection_status")
                == "POTENTIAL_NEW_UNKNOWN"
            and evidence.get("confidence_route")
                == "LOW_CONFIDENCE"
            and evidence.get("kb_matched") is False
            and evidence.get("ae_anomalous") is True
        ):
            raise ValueError(
                "Candidate evidence does not satisfy "
                "promotion eligibility"
            )

        vector = self.transform_flow(
            candidate["flow"]
        )

        # Compare against frozen KB using the saved index.
        frozen_distances, _ = (
            self.frozen_index.kneighbors(
                vector.reshape(1, -1),
                n_neighbors=1
            )
        )

        frozen_distance = float(
            frozen_distances[0, 0]
        )

        if frozen_distance <= self.cutoff:
            raise ValueError(
                "Candidate already matches frozen KB; "
                "promotion rejected"
            )

        existing_patterns = self._load_patterns()

        for pattern in existing_patterns:
            distance = float(
                np.linalg.norm(
                    vector - pattern["vector"]
                )
            )

            if distance <= NUMERIC_TOLERANCE:
                raise ValueError(
                    "Duplicate writable KB pattern"
                )

        digest = self.vector_hash(vector)
        pattern_id = "WKB-" + digest[:16]

        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO promoted_patterns (
                    pattern_id,
                    candidate_id,
                    vector_json,
                    vector_hash,
                    promoted_at,
                    promoted_by,
                    promotion_note,
                    frozen_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pattern_id,
                    candidate_id,
                    json.dumps(
                        vector.tolist(),
                        allow_nan=False
                    ),
                    digest,
                    utc_now(),
                    promoted_by.strip(),
                    promotion_note.strip(),
                    self.frozen_hash,
                )
            )

        return {
            "pattern_id": pattern_id,
            "candidate_id": candidate_id,
            "frozen_distance": frozen_distance,
            "status": "PROMOTED",
            "mode": "SHADOW",
        }

    def lookup(self, flow):
        """
        Search promoted patterns only.

        Does not modify the existing GC2 routing.
        """

        vector = self.transform_flow(flow)
        patterns = self._load_patterns()

        if not patterns:
            return {
                "matched": False,
                "nearest_pattern_id": None,
                "nearest_distance": None,
                "cutoff": self.cutoff,
                "mode": "SHADOW",
            }

        vectors = np.vstack([
            item["vector"]
            for item in patterns
        ])

        distances = np.linalg.norm(
            vectors - vector,
            axis=1
        )

        nearest = int(np.argmin(distances))
        distance = float(distances[nearest])

        return {
            "matched": distance <= self.cutoff,
            "nearest_pattern_id": (
                patterns[nearest]["pattern_id"]
            ),
            "nearest_distance": distance,
            "cutoff": self.cutoff,
            "mode": "SHADOW",
        }

    def list_promoted(self):
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT pattern_id, candidate_id,
                       promoted_at, promoted_by,
                       promotion_note
                FROM promoted_patterns
                ORDER BY promoted_at
                """
            ).fetchall()

        return [dict(row) for row in rows]
