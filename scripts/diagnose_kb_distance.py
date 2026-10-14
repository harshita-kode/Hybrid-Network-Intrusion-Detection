
"""Diagnose KNN distance reconstruction mismatch."""

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

MANIFEST = (
    ROOT / "outputs/results/gc2_evaluation"
    / "unknown_detection/evaluation_manifest.csv"
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


def main():
    print("Loading evaluation manifest...")

    manifest = pd.read_csv(MANIFEST)
    first = manifest.iloc[0]

    print("Source split:", first["source_split"])
    print("Source row:", first["source_row"])

    source = pd.read_parquet(
        DATA_DIR / f"{first['source_split']}.parquet"
    )

    flow = source.iloc[
        [int(first["source_row"])]
    ].copy()

    print("Loading KNN artifact...")

    pipeline = GC2Pipeline()
    artifact = pipeline.kb_artifact

    query = np.asarray(
        transform(flow, artifact),
        dtype=np.float64
    )[0]

    distances, indices = artifact["index"].kneighbors(
        query.reshape(1, -1),
        n_neighbors=1
    )

    index_distance = float(distances[0, 0])
    nearest_index = int(indices[0, 0])

    neighbor = np.asarray(
        artifact["index"]._fit_X[nearest_index],
        dtype=np.float64
    )

    differences = query - neighbor

    reproduced = float(
        np.linalg.norm(differences)
    )

    absolute_error = abs(
        reproduced - index_distance
    )

    relative_error = (
        absolute_error / max(
            reproduced,
            index_distance,
            1e-12
        )
    )

    print("\n========== KNN DISTANCE DIAGNOSTIC ==========")

    print("Query shape:", query.shape)
    print("Neighbour shape:", neighbor.shape)

    print("Query finite:", np.isfinite(query).all())
    print("Neighbour finite:", np.isfinite(neighbor).all())

    print("Nearest index:", nearest_index)

    print("KNN index distance:", repr(index_distance))
    print("Reconstructed distance:", repr(reproduced))

    print("Absolute error:", repr(absolute_error))
    print("Relative error:", repr(relative_error))

    print(
        "Original tolerance passed:",
        np.isclose(
            reproduced,
            index_distance,
            rtol=1e-5,
            atol=1e-7
        )
    )

    print(
        "Relaxed diagnostic tolerance passed:",
        np.isclose(
            reproduced,
            index_distance,
            rtol=1e-5,
            atol=1e-6
        )
    )

    print("\n============================================")


if __name__ == "__main__":
    main()
