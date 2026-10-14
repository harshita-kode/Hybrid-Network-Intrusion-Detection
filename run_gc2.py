
"""
GC2 - Single-file intrusion detection entry point.

Stage 1:
XGBoost + SHAP + Security Alert + JSON

Usage:
    python run_gc2.py --input path/to/traffic.csv
    python run_gc2.py --input path/to/traffic.parquet
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.pipeline.integration import GC2Pipeline
from src.pipeline.alert import (
    alert_to_dict,
    format_alert,
)


ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT / "outputs" / "alerts"


def load_input(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(
        "Input must be a CSV or Parquet file"
    )


def safe_filename(value):
    return re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        str(value)
    )


def main():
    parser = argparse.ArgumentParser(
        description="GC2 Network Intrusion Detection"
    )

    parser.add_argument(
        "--input",
        required=False,
        help="Incoming network-flow CSV or Parquet file"
    )

    parser.add_argument(
        "--known-test-row",
        type=int,
        default=None,
        help="Original known_test.parquet row for AE evaluation"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of flows to process"
    )

    args = parser.parse_args()
    
    if args.known_test_row is None and not args.input:
        parser.error(
            "Provide --input or --known-test-row"
        )

    if args.known_test_row is not None and args.input:
        parser.error(
            "Use --known-test-row or --input, not both"
        )

    if args.known_test_row is not None and args.limit != 1:
        parser.error(
            "--known-test-row supports one flow per run"
        )

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    print("\n========== GC2 STARTING ==========")

    history = None

    if args.known_test_row is not None:
        from src.pipeline.ae_context import KnownTestContext

        history = KnownTestContext()

        if not 0 <= args.known_test_row < len(history.data):
            raise IndexError("Known-test row out of range")

        data = history.data.iloc[
            [args.known_test_row]
        ].copy()
    else:
        data = load_input(args.input)

    if data.empty:
        raise ValueError("Input dataset is empty")

    pipeline = GC2Pipeline()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total = min(args.limit, len(data))

    print(f"Processing {total} network flows...")

    for index in range(total):
        flow = data.iloc[[index]]

        alert_id = f"GC2-{index + 1:06d}"

                
        prepared_ae_flow = None
        historical_flow_count = None

        if history is not None:
            # Prepare context only if XGBoost is low confidence
            # and the frozen KB reports a miss.
            features = flow[pipeline.feature_names]
            probabilities = pipeline.model.predict_proba(
                features
            )[0]

            confidence = float(probabilities.max())

            if confidence < pipeline.confidence_threshold:
                from notebooks.kb.kb_flow import search

                lookup = search(
                    flow,
                    pipeline.kb_artifact,
                    pipeline.kb_reference,
                    k=5,
                    trace=False
                )[0]

                if not lookup["matched"]:
                    prepared_ae_flow, historical_flow_count = (
                        history.prepare(args.known_test_row)
                    )

        alert = pipeline.predict_one(
            flow=flow,
            alert_id=alert_id,
            prepared_ae_flow=prepared_ae_flow,
            historical_flow_count=historical_flow_count,
        )


        print("\n" + format_alert(alert))

        output_path = (
            OUTPUT_DIR
            / f"{safe_filename(alert_id)}.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                alert_to_dict(alert),
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False
            )

        print(
            f"\nSaved JSON: {output_path}"
        )

    print("\n========== GC2 COMPLETE ==========")


if __name__ == "__main__":
    main()
