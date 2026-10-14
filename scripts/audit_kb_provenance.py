
"""
GC2 Stage 10E — Frozen KB provenance audit.

Read-only checks:
1. Inspect KB reference metadata.
2. Inspect reference labels and identifiers.
3. Compare reference identities against evaluation splits.
4. Check exact transformed-vector overlap.
5. Report KB similarity-match rates.

Does not modify the frozen KB or datasets.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.integration import GC2Pipeline
from notebooks.kb.kb_flow import transform


DATA_DIR = (
    ROOT / "data/processed/Data_V3_SSH_Unknown_final"
)

OUTPUT_DIR = (
    ROOT / "outputs/results/gc2_evaluation/kb_provenance"
)

KEY_COLUMNS = [
    "Flow ID",
    "Timestamp",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
]

SAMPLE_SIZE = 500
RANDOM_STATE = 42


def normalized_keys(frame, columns):
    """
    Produce deterministic row hashes for the supplied
    identity columns.
    """

    return pd.util.hash_pandas_object(
        frame[columns],
        index=False
    ).to_numpy(dtype=np.uint64)


def inspect_reference(reference):
    print("\n========== KB REFERENCE INSPECTION ==========")

    print("Reference type:", type(reference).__name__)
    print("Reference records:", len(reference))

    if isinstance(reference, pd.DataFrame):
        print("Columns:", reference.columns.tolist())
        return reference.copy()

    if isinstance(reference, list):
        frame = pd.DataFrame(reference)
        print("Columns:", frame.columns.tolist())
        return frame

    if isinstance(reference, dict):
        print("Dictionary keys:", list(reference.keys()))

        try:
            frame = pd.DataFrame(reference)
        except (ValueError, TypeError):
            return None

        if len(frame) == len(reference):
            return frame

    return None


def inspect_labels(reference_frame):
    if reference_frame is None:
        print("Reference metadata unavailable.")
        return {}

    label_column = next(
        (
            name for name in (
                "Label",
                "label",
                "Original_Label",
                "attack_type",
            )
            if name in reference_frame.columns
        ),
        None
    )

    if label_column is None:
        print("No reference label column found.")
        return {}

    counts = (
        reference_frame[label_column]
        .astype(str)
        .value_counts()
    )

    print("\n========== KB REFERENCE LABELS ==========")
    print(counts.to_string())

    return {
        str(k): int(v)
        for k, v in counts.items()
    }


def inspect_identity_overlap(reference_frame):
    """
    Check exact flow identity overlap where sufficient
    reference metadata is available.
    """

    results = {}

    if reference_frame is None:
        print("Identity overlap unavailable.")
        return results

    available = [
        column for column in KEY_COLUMNS
        if column in reference_frame.columns
    ]

    if len(available) != len(KEY_COLUMNS):
        print(
            "Full flow identity unavailable. "
            "Missing:",
            sorted(set(KEY_COLUMNS) - set(available))
        )
        return results

    reference_hashes = normalized_keys(
        reference_frame,
        KEY_COLUMNS
    )

    for split in ("known_test", "unknown_test"):
        test = pd.read_parquet(
            DATA_DIR / f"{split}.parquet",
            columns=KEY_COLUMNS
        )

        test_hashes = normalized_keys(
            test,
            KEY_COLUMNS
        )

        overlap = int(
            np.isin(
                test_hashes,
                reference_hashes
            ).sum()
        )

        results[split] = {
            "records": len(test),
            "exact_identity_overlap": overlap,
        }

        print(
            f"{split}: {overlap} exact identity overlaps"
        )

    return results


def inspect_vector_overlap(pipeline):
    """
    Compare selected evaluation records against the
    frozen KB in the saved transformed feature space.

    This is a diagnostic sample, not a full-dataset
    overlap count.
    """

    artifact = pipeline.kb_artifact

    reference_vectors = np.asarray(
        artifact["vectors"],
        dtype=np.float64
    )

    results = {}

    for split in ("known_test", "unknown_test"):
        data = pd.read_parquet(
            DATA_DIR / f"{split}.parquet"
        )

        sample = data.sample(
            n=min(SAMPLE_SIZE, len(data)),
            random_state=RANDOM_STATE
        )

        queries = np.asarray(
            transform(sample, artifact),
            dtype=np.float64
        )

        if not np.isfinite(queries).all():
            raise ValueError(
                "Nonfinite transformed query vectors"
            )

        distances, indices = (
            artifact["index"].kneighbors(
                queries,
                n_neighbors=1
            )
        )

        # Reconstruct distances directly to avoid
        # numerical near-zero artefacts from optimized
        # nearest-neighbour calculations.
        neighbours = reference_vectors[
            indices[:, 0]
        ]

        direct_distances = np.linalg.norm(
            queries - neighbours,
            axis=1
        )

        exact_vector_matches = int(
            np.count_nonzero(
                np.all(
                    queries == neighbours,
                    axis=1
                )
            )
        )

        near_zero_matches = int(
            np.count_nonzero(
                direct_distances <= 1e-6
            )
        )

        kb_matches = int(
            np.count_nonzero(
                distances[:, 0] <= artifact["cutoff"]
            )
        )

        results[split] = {
            "sample_size": len(sample),
            "exact_vector_matches": exact_vector_matches,
            "near_zero_vector_matches": near_zero_matches,
            "kb_similarity_matches": kb_matches,
            "median_nearest_distance": float(
                np.median(distances[:, 0])
            ),
        }

        print(f"\n========== {split.upper()} ==========")
        print("Sample size:", len(sample))
        print("Exact vector matches:", exact_vector_matches)
        print("Near-zero vector matches:", near_zero_matches)
        print("KB similarity matches:", kb_matches)
        print(
            "Median nearest distance:",
            results[split]["median_nearest_distance"]
        )

    return results


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading frozen GC2 Knowledge Base...")

    pipeline = GC2Pipeline()

    artifact = pipeline.kb_artifact

    reference_frame = inspect_reference(
        pipeline.kb_reference
    )

    if reference_frame is not None:
        if len(reference_frame) != artifact["count"]:
            raise ValueError(
                "KB reference count does not match artifact"
            )

    label_counts = inspect_labels(
        reference_frame
    )

    identity_overlap = inspect_identity_overlap(
        reference_frame
    )

    vector_overlap = inspect_vector_overlap(
        pipeline
    )

    report = {
        "kb_records": int(artifact["count"]),
        "kb_features": len(artifact["columns"]),
        "kb_cutoff": float(artifact["cutoff"]),
        "reference_labels": label_counts,
        "identity_overlap": identity_overlap,
        "vector_overlap": vector_overlap,
        "limitations": [
            "Exact identity overlap requires full reference metadata.",
            "Vector-overlap results use 500 sampled flows per split.",
            "Feature-vector equality does not necessarily mean identical original records.",
            "KB construction provenance must be confirmed from the original build script.",
        ],
    }

    output_path = OUTPUT_DIR / "kb_provenance_audit.json"

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            allow_nan=False
        )

    print("\n========== KB PROVENANCE AUDIT ==========")
    print("KB records:", report["kb_records"])
    print("KB features:", report["kb_features"])
    print("Audit saved:", output_path)

    print("\nKB PROVENANCE AUDIT COMPLETE")


if __name__ == "__main__":
    main()
