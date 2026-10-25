
"""
GC2 Stage 10D.3
Independent frozen KNN + Autoencoder evaluation.

Uses 180 selected flows:
- 90 high-confidence benign predictions on unknown attacks
- 90 benign controls

Uses causal behavioural features computed once on
clean_master in stable chronological order.

Does not modify models, KBs, candidates, or production routing.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.integration import (
    GC2Pipeline,
    AutoencoderIntegration,
    explain_kb_match,
)
from src.pipeline.incoming_context import add_behavioural_features


DATA_DIR = (
    ROOT / "data/processed/Data_V3_SSH_Unknown_final"
)

OUTPUT_DIR = (
    ROOT / "outputs/results/gc2_evaluation/unknown_detection"
)

MANIFEST_PATH = OUTPUT_DIR / "evaluation_manifest.csv"

KEY_COLUMNS = [
    "Flow ID",
    "Timestamp",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
]

LABEL_COLUMNS = [
    "Label",
    "Original_Label",
    "Attempted Category",
    "is_attempted",
    "source_file",
]


def row_keys(frame):
    return pd.util.hash_pandas_object(
        frame[KEY_COLUMNS],
        index=False
    ).to_numpy(dtype=np.uint64)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start_total = time.perf_counter()

    print("Loading evaluation manifest...")
    manifest = pd.read_csv(MANIFEST_PATH)

    if len(manifest) != 180:
        raise ValueError(
            f"Expected 180 evaluation flows, got {len(manifest)}"
        )

    print("Loading master dataset...")
    master = pd.read_parquet(
        DATA_DIR / "clean_master.parquet"
    )

    # Preserve original row order for equal timestamps.
    master = master.reset_index(drop=True)

    print("Locating evaluation flows...")

    master_hashes = row_keys(master)

    key_positions = {}

    for position, key in enumerate(master_hashes):
        key = int(key)

        if key in key_positions:
            key_positions[key] = None
        else:
            key_positions[key] = position

    target_positions = []

    source_cache = {}

    for row in manifest.itertuples(index=False):

        split = row.source_split

        if split not in source_cache:
            source_cache[split] = pd.read_parquet(
                DATA_DIR / f"{split}.parquet"
            )

        source = source_cache[split]

        target = source.iloc[int(row.source_row)]

        key = int(
            row_keys(
                pd.DataFrame([target])
            )[0]
        )

        position = key_positions.get(key)

        if position is None:
            raise ValueError(
                "Missing or ambiguous master flow: "
                f"{split} row {row.source_row}"
            )

        if str(target["Label"]) != str(row.actual_label):
            raise ValueError(
                "Manifest ground truth does not match source"
            )

        target_positions.append(position)

    if len(set(target_positions)) != len(target_positions):
        raise ValueError(
            "Evaluation targets contain duplicate master records"
        )

    print("All evaluation flows uniquely located.")

    # -------------------------------------------------
    # Calculate behavioural features once.
    # -------------------------------------------------

    print("Preparing chronological behavioural features...")

    context_start = time.perf_counter()

    context_input = master.drop(
        columns=LABEL_COLUMNS,
        errors="ignore"
    )

    prepared_master = add_behavioural_features(
        context_input
    )

    context_seconds = (
        time.perf_counter() - context_start
    )

    print(
        f"Behavioural preparation: {context_seconds:.2f}s"
    )

    # -------------------------------------------------
    # Load frozen detection components.
    # -------------------------------------------------

    print("Loading frozen KNN and Autoencoder...")

    pipeline = GC2Pipeline()

    ae = AutoencoderIntegration()

    # -------------------------------------------------
    # Evaluate each selected record.
    # -------------------------------------------------

    rows = []

    for number, (manifest_row, position) in enumerate(
        zip(manifest.itertuples(index=False), target_positions),
        start=1
    ):

        flow = master.iloc[[position]].copy()

        prepared_flow = (
            prepared_master.iloc[position].to_dict()
        )

        started = time.perf_counter()

        kb_result = explain_kb_match(
            flow=flow,
            artifact=pipeline.kb_artifact,
            reference=pipeline.kb_reference,
            top_n=5
        )

        kb_matched = bool(
            kb_result["kb_matched"]
        )

        ae_called = False
        ae_anomalous = None
        ae_mse = None
        ae_threshold = None
        mse_consistency = None

        if not kb_matched:

            ae_called = True

            ae_result, ae_explanation = ae.analyze(
                prepared_flow
            )

            ae_anomalous = bool(
                ae_result["anomalous"]
            )

            ae_mse = float(
                ae_result["anomaly_mse"]
            )

            ae_threshold = float(
                ae_result["threshold"]
            )

            mse_consistency = bool(
                ae_result["mse_consistency"]
            )

            if not mse_consistency:
                raise ValueError(
                    "Autoencoder MSE consistency failed"
                )

        # A KB match is not automatically a malicious
        # detection because KB provenance is unverified.
        #
        # Count only AE-confirmed anomalies as
        # suspicious in this conservative experiment.

        suspicious = (
            ae_called
            and ae_anomalous is True
        )

        elapsed = time.perf_counter() - started

        rows.append({
            "source_split": manifest_row.source_split,
            "source_row": int(manifest_row.source_row),
            "actual_label": manifest_row.actual_label,
            "selection_group": manifest_row.selection_group,
            "master_position": int(position),
            "kb_matched": kb_matched,
            "kb_distance": float(
                kb_result["nearest_distance"]
            ),
            "kb_cutoff": float(
                kb_result["match_cutoff"]
            ),
            "nearest_kb_pattern": (
                kb_result["nearest_pattern_id"]
            ),
            "ae_called": ae_called,
            "ae_anomalous": ae_anomalous,
            "ae_mse": ae_mse,
            "ae_threshold": ae_threshold,
            "mse_consistency": mse_consistency,
            "experimental_suspicious": bool(suspicious),
            "processing_seconds": elapsed,
        })

        print(
            f"[{number:03d}/180] "
            f"{manifest_row.actual_label} | "
            f"KB={'MATCH' if kb_matched else 'MISS'} | "
            f"AE={ae_anomalous} | "
            f"Suspicious={suspicious}"
        )

    results = pd.DataFrame(rows)

    results.to_csv(
        OUTPUT_DIR / "independent_predictions.csv",
        index=False
    )

    # -------------------------------------------------
    # Aggregate results.
    # -------------------------------------------------

    results["is_unknown_attack"] = (
        results["actual_label"] != "BENIGN"
    )

    unknown = results[
        results["is_unknown_attack"]
    ]

    benign = results[
        ~results["is_unknown_attack"]
    ]

    per_class = (
        results.groupby("actual_label")
        .agg(
            samples=("actual_label", "size"),
            kb_matches=("kb_matched", "sum"),
            ae_calls=("ae_called", "sum"),
            suspicious_count=(
                "experimental_suspicious",
                "sum"
            ),
            mean_processing_seconds=(
                "processing_seconds",
                "mean"
            ),
        )
        .reset_index()
    )

    per_class["suspicious_rate"] = (
        per_class["suspicious_count"]
        / per_class["samples"]
    )

    per_class["kb_match_rate"] = (
        per_class["kb_matches"]
        / per_class["samples"]
    )

    per_class.to_csv(
        OUTPUT_DIR / "per_attack_results.csv",
        index=False
    )

    total_seconds = (
        time.perf_counter() - start_total
    )

    metrics = {
        "samples": int(len(results)),
        "unknown_samples": int(len(unknown)),
        "benign_samples": int(len(benign)),
        "unknown_suspicious_count": int(
            unknown["experimental_suspicious"].sum()
        ),
        "unknown_suspicious_rate": float(
            unknown["experimental_suspicious"].mean()
        ),
        "benign_false_positive_count": int(
            benign["experimental_suspicious"].sum()
        ),
        "benign_false_positive_rate": float(
            benign["experimental_suspicious"].mean()
        ),
        "kb_match_count": int(
            results["kb_matched"].sum()
        ),
        "ae_call_count": int(
            results["ae_called"].sum()
        ),
        "context_preparation_seconds": float(
            context_seconds
        ),
        "total_evaluation_seconds": float(
            total_seconds
        ),
        "evaluation_type": (
            "Selected high-confidence benign unknown attacks "
            "versus sampled benign controls"
        ),
        "limitations": [
            "Not representative of all unknown-test traffic.",
            "KB matches are not automatically counted malicious.",
            "Uses master chronology, not live arrival order.",
            "Master history may differ from AE calibration context.",
            "Unknown-test independence from AE and KB training must be verified.",
            "XGBoost leakage audit remains unresolved.",
        ],
    }

    with (
        OUTPUT_DIR / "independent_metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            indent=2,
            allow_nan=False
        )

    print("\n========== INDEPENDENT EVALUATION ==========")

    print(
        "Unknown suspicious rate:",
        f"{metrics['unknown_suspicious_rate']:.2%}"
    )

    print(
        "Benign false-positive rate:",
        f"{metrics['benign_false_positive_rate']:.2%}"
    )

    print("KB matches:", metrics["kb_match_count"])
    print("AE calls:", metrics["ae_call_count"])

    print("\nPer-class results:")
    print(per_class.to_string(index=False))

    print("\nResults saved to:", OUTPUT_DIR)
    print("\nINDEPENDENT EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
