
"""GC2 Stage 12D: Autoencoder evaluation on frozen-KB misses.

Read-only experiment:
- 58 held-out unknown KB misses (expected from Stage 12B)
- 1,301 sampled benign KB misses
- Causal behavioural features from clean_master
- Frozen Version20 Autoencoder

Does not modify KB, models, or production decisions.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.integration import AutoencoderIntegration
from src.pipeline.incoming_context import add_behavioural_features


DATA_DIR = ROOT / "data/processed/Data_V3_SSH_Unknown_final"
RESULT_DIR = ROOT / "outputs/results/gc2_evaluation/stage12"

KEY_COLUMNS = [
    "Flow ID", "Timestamp", "Src IP", "Src Port",
    "Dst IP", "Dst Port", "Protocol",
]

DROP_COLUMNS = [
    "Label", "Original_Label", "Attempted Category",
    "is_attempted", "source_file",
]


def make_keys(frame):
    return pd.util.hash_pandas_object(
        frame[KEY_COLUMNS], index=False
    ).to_numpy(dtype=np.uint64)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    print("Loading Stage 12B predictions...")
    predictions = pd.read_csv(
        RESULT_DIR / "kb_retrieval_predictions.csv"
    )

    print("Loading held-out unknown and benign datasets...")
    unknown = pd.read_parquet(
        DATA_DIR / "unknown_kb_splits/unknown_final_test.parquet"
    )

    known = pd.read_parquet(
        DATA_DIR / "known_test.parquet"
    )

    benign = known.loc[
        known["Label"] == "BENIGN"
    ].sample(
        n=len(unknown), random_state=42
    )

    evaluation = pd.concat(
        [unknown, benign],
        ignore_index=True
    )

    if len(evaluation) != len(predictions):
        raise ValueError("Stage 12B sample count mismatch")

    # Stage 12B concatenated unknown first, benign second.
    if not np.array_equal(
        evaluation["Label"].astype(str).to_numpy(),
        predictions["actual_label"].astype(str).to_numpy()
    ):
        raise ValueError("Evaluation labels do not align")

    # Keep only frozen KB misses.
    miss_positions = np.flatnonzero(
        ~predictions["kb_matched"].astype(bool).to_numpy()
    )

    targets = evaluation.iloc[miss_positions].copy()

    print("Total KB misses:", len(targets))
    print(targets["Label"].value_counts().to_string())

    print("\nLoading clean master history...")
    master = pd.read_parquet(
        DATA_DIR / "clean_master.parquet"
    ).reset_index(drop=True)

    print("Mapping targets to original master positions...")

    master_keys = make_keys(master)
    target_keys = make_keys(targets)

    positions = {}

    for i, key in enumerate(master_keys):
        key = int(key)
        if key in positions:
            positions[key] = None
        else:
            positions[key] = i

    target_positions = []

    for key in target_keys:
        position = positions.get(int(key))
        if position is None:
            raise ValueError(
                "Target missing or ambiguous in master history"
            )
        target_positions.append(position)

    if len(set(target_positions)) != len(target_positions):
        raise ValueError("Duplicate evaluation targets")

    print("Master record mapping: PASSED")

    print("\nPreparing causal behavioural features...")
    context_start = time.perf_counter()

    prepared_master = add_behavioural_features(
        master.drop(columns=DROP_COLUMNS, errors="ignore")
    )

    context_seconds = time.perf_counter() - context_start

    print(
        f"Behavioural feature preparation: "
        f"{context_seconds:.2f} seconds"
    )

    print("\nLoading Version20 Autoencoder...")
    ae = AutoencoderIntegration()

    results = []

    for number, (position, (_, target)) in enumerate(
        zip(target_positions, targets.iterrows()),
        start=1
    ):
        prepared = prepared_master.iloc[position].to_dict()

        started = time.perf_counter()
        result, _ = ae.analyze(prepared)
        elapsed = time.perf_counter() - started

        if not result["mse_consistency"]:
            raise ValueError("Autoencoder MSE verification failed")

        results.append({
            "actual_label": str(target["Label"]),
            "master_position": int(position),
            "anomalous": bool(result["anomalous"]),
            "anomaly_mse": float(result["anomaly_mse"]),
            "threshold": float(result["threshold"]),
            "processing_seconds": elapsed,
        })

        if number % 50 == 0 or number == len(targets):
            print(f"Evaluated {number}/{len(targets)} flows")

    results = pd.DataFrame(results)

    results.to_csv(
        RESULT_DIR / "ae_predictions.csv",
        index=False
    )

    per_class = (
        results.groupby("actual_label")
        .agg(
            samples=("anomalous", "size"),
            anomalies=("anomalous", "sum"),
            mean_mse=("anomaly_mse", "mean"),
            median_mse=("anomaly_mse", "median"),
        )
        .reset_index()
    )

    per_class["anomaly_rate"] = (
        per_class["anomalies"] / per_class["samples"]
    )

    per_class.to_csv(
        RESULT_DIR / "ae_per_class.csv",
        index=False
    )

    unknown_results = results.loc[
        results["actual_label"] != "BENIGN"
    ]
    benign_results = results.loc[
        results["actual_label"] == "BENIGN"
    ]

    metrics = {
        "unknown_kb_misses": int(len(unknown_results)),
        "benign_kb_misses": int(len(benign_results)),
        "unknown_ae_anomalies": int(
            unknown_results["anomalous"].sum()
        ),
        "unknown_ae_anomaly_rate": float(
            unknown_results["anomalous"].mean()
        ),
        "benign_ae_anomalies": int(
            benign_results["anomalous"].sum()
        ),
        "benign_ae_false_positive_rate": float(
            benign_results["anomalous"].mean()
        ),
        "context_seconds": context_seconds,
        "total_seconds": time.perf_counter() - start,
        "evaluation_scope": (
            "Frozen KB misses only, using master chronology"
        ),
    }

    with (
        RESULT_DIR / "ae_metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, allow_nan=False)

    print("\n========== STAGE 12D RESULTS ==========")
    print(
        "Unknown AE anomaly rate:",
        f"{metrics['unknown_ae_anomaly_rate']:.2%}"
    )
    print(
        "Benign AE false-positive rate:",
        f"{metrics['benign_ae_false_positive_rate']:.2%}"
    )
    print("\nPer-class results:")
    print(per_class.to_string(index=False))

    print("\nSTAGE 12D COMPLETE")


if __name__ == "__main__":
    main()
