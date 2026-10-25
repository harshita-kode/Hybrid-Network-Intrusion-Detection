
"""GC2 Stage 12B: frozen KB retrieval evaluation.

Read-only evaluation on:
- All held-out unknown_final_test flows
- Reproducible benign controls from known_test

Does not modify models or KB records.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from notebooks.kb.kb_flow import transform


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT / "data/processed/Data_V3_SSH_Unknown_final"
)

SPLIT_DIR = DATA_DIR / "unknown_kb_splits"

KB_PATH = (
    ROOT / "saved-models/knn_kb/initial_kb.joblib"
)

OUTPUT_DIR = (
    ROOT / "outputs/results/gc2_evaluation/stage12"
)


def evaluate_group(frame, artifact, group):
    started = time.perf_counter()

    vectors = np.asarray(
        transform(frame, artifact),
        dtype=np.float64,
    )

    if vectors.shape != (
        len(frame),
        len(artifact["columns"]),
    ):
        raise ValueError("Transformed feature shape mismatch")

    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite transformed features")

    distances, indices = artifact["index"].kneighbors(
        vectors,
        n_neighbors=1,
    )

    nearest_distances = distances[:, 0]

    # Use the frozen KB's actual saved cutoff.
    matches = (
        nearest_distances <= float(artifact["cutoff"])
    )

    elapsed = time.perf_counter() - started

    return pd.DataFrame({
        "group": group,
        "actual_label": frame["Label"].to_numpy(),
        "kb_matched": matches.astype(bool),
        "nearest_distance": nearest_distances,
        "nearest_index": indices[:, 0],
        "kb_cutoff": float(artifact["cutoff"]),
    }), elapsed


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading frozen KB...")
    artifact = joblib.load(KB_PATH)

    print("Loading final unknown-test partition...")
    unknown = pd.read_parquet(
        SPLIT_DIR / "unknown_final_test.parquet"
    )

    print("Loading benign controls...")
    known = pd.read_parquet(
        DATA_DIR / "known_test.parquet"
    )

    benign = known.loc[
        known["Label"] == "BENIGN"
    ].sample(
        n=len(unknown),
        random_state=42,
    )

    print("Unknown samples:", len(unknown))
    print("Benign samples:", len(benign))

    unknown_results, unknown_time = evaluate_group(
        unknown, artifact, "UNKNOWN"
    )

    benign_results, benign_time = evaluate_group(
        benign, artifact, "BENIGN"
    )

    results = pd.concat(
        [unknown_results, benign_results],
        ignore_index=True,
    )

    results.to_csv(
        OUTPUT_DIR / "kb_retrieval_predictions.csv",
        index=False,
    )

    per_class = (
        results.groupby(["group", "actual_label"])
        .agg(
            samples=("kb_matched", "size"),
            matches=("kb_matched", "sum"),
            median_distance=("nearest_distance", "median"),
            mean_distance=("nearest_distance", "mean"),
        )
        .reset_index()
    )

    per_class["match_rate"] = (
        per_class["matches"] / per_class["samples"]
    )

    per_class.to_csv(
        OUTPUT_DIR / "kb_retrieval_per_class.csv",
        index=False,
    )

    metrics = {
        "unknown_samples": int(len(unknown)),
        "benign_samples": int(len(benign)),
        "unknown_kb_match_count": int(
            unknown_results["kb_matched"].sum()
        ),
        "unknown_kb_match_rate": float(
            unknown_results["kb_matched"].mean()
        ),
        "benign_kb_match_count": int(
            benign_results["kb_matched"].sum()
        ),
        "benign_kb_match_rate": float(
            benign_results["kb_matched"].mean()
        ),
        "unknown_lookup_seconds": unknown_time,
        "benign_lookup_seconds": benign_time,
        "kb_cutoff": float(artifact["cutoff"]),
        "evaluation_type": (
            "Frozen KB retrieval on held-out same-family "
            "unknown flows and sampled benign controls"
        ),
    }

    with (
        OUTPUT_DIR / "kb_retrieval_metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            indent=2,
            allow_nan=False,
        )

    print("\n========== STAGE 12B RESULTS ==========")
    print(
        "Unknown KB match rate:",
        f"{metrics['unknown_kb_match_rate']:.2%}",
    )
    print(
        "Benign KB match rate:",
        f"{metrics['benign_kb_match_rate']:.2%}",
    )

    print("\nPer-class results:")
    print(per_class.to_string(index=False))

    print("\nSaved results to:", OUTPUT_DIR)
    print("\nSTAGE 12B COMPLETE")


if __name__ == "__main__":
    main()
