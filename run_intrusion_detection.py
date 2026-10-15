
"""
GC2 - Single-file intrusion detection entry point.

Stage 1:
XGBoost + SHAP + Security Alert + JSON

Usage:
    python run_intrusion_detection.py --input path/to/traffic.csv
    python run_intrusion_detection.py --input path/to/traffic.parquet
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from datetime import datetime, timezone
from uuid import uuid4

from src.pipeline.integration import GC2Pipeline
from src.pipeline.alert import (
    alert_to_dict,
    format_alert,
)
from src.pipeline.kb_candidates import KBCandidateStore

from src.pipeline.kb_promotion import WritableKB
from src.analysis.kb_explanation import explain_kb_match
from src.pipeline.integration import AutoencoderIntegration
from src.pipeline.hybrid_decision import resolve_hybrid_decision


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
        description="Network Intrusion Detection"
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

    parser.add_argument(
        "--asset-id",
        type=str,
        default=None,
        help="Explicit asset ID for illustrative risk assessment",
    )

    parser.add_argument(
        "--acceptable-risk",
        type=float,
        default=None,
        help="Configured acceptable-risk score (0 to 192)",
    )

    parser.add_argument(
        "--history",
        type=str,
        default=None,
        help="Chronological history CSV or Parquet for AE context"
    )

    parser.add_argument(
        "--store-candidates",
        action="store_true",
        help="Store eligible unknown patterns for analyst review"
    )
    parser.add_argument(
        "--shadow-kb",
        action="store_true",
        help="Run writable KB lookup as supporting evidence only"
    )
    parser.add_argument(
        "--hybrid-screening",
        action="store_true",
        help="Screen high-confidence benign flows using frozen KB and AE in shadow mode",
    )

    parser.add_argument(
        "--demo",
        choices=("known", "benign", "unknown", "kb-match", "kb-miss"),
        help="Run a verified built-in demonstration scenario",
    )

    args = parser.parse_args()

    if args.demo is not None:
        if args.input or args.known_test_row is not None:
            parser.error("--demo cannot be combined with --input or --known-test-row")
        if args.history:
            parser.error("--demo cannot be combined with --history")
        if args.limit != 1:
            parser.error("--demo processes exactly one flow")

        if args.asset_id is None:
            args.asset_id = "A001"
        if args.acceptable_risk is None:
            args.acceptable_risk = 100.0

        if args.demo == "known":
            args.known_test_row = 4
        elif args.demo == "benign":
            args.known_test_row = 0
        elif args.demo in ("unknown", "kb-miss"):
            args.known_test_row = 235964
        elif args.demo == "kb-match":
            # Verified against real XGBoost + frozen KB: low-confidence
            # DoS Slowhttptest flow, KB distance about 0.023755.
            args.input = str(
                ROOT / "data" / "processed"
                / "Data_V3_SSH_Unknown_final" / "unknown_kb_splits"
                / "unknown_final_test.parquet"
            )

        print(f"\nGC2 demo selected: {args.demo}")

    
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


    if args.history and args.known_test_row is not None:
        parser.error(
            "--history cannot be combined with --known-test-row"
        )

    if args.history and not args.input:
        parser.error(
            "--history requires --input"
        )
    print("\n========== STARTING ==========")

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

    
    # =====================================================
    # LOAD INCOMING CHRONOLOGICAL HISTORY
    # =====================================================

    incoming_context = None

    if args.history is not None:

        from src.pipeline.incoming_context import IncomingContext

        history_data = load_input(args.history)

        incoming_context = IncomingContext(
            history_data
        )

        print(
            f"Loaded {len(history_data):,} historical flows"
        )


    if args.demo == "kb-match":
        if len(data) <= 389:
            raise ValueError("KB-match demo dataset must contain row 389")
        data = data.iloc[[389]].copy()
        record = data.iloc[0]
        if not (
            str(record.get("Label")) == "DoS Slowhttptest"
            and str(record.get("source_file")) == "wednesday"
            and int(record.get("id")) == 30028
        ):
            raise ValueError(
                "KB-match demo record identity changed; refusing to use an unverified row"
            )
        data = data.drop(
            columns=["Label", "Original_Label", "Attempted Category", "is_attempted"],
            errors="ignore",
        )
        print("KB-match demo: final-test row 389, original ID 30028")

    if data.empty:
        raise ValueError("Input dataset is empty")

    pipeline = GC2Pipeline()

    shadow_ae = None

    if args.hybrid_screening:
        print("Hybrid shadow screening enabled.")
        print("Production decisions will remain unchanged.")

    candidate_store = None

    if args.store_candidates:
        candidate_store = KBCandidateStore()
        print("KB candidate storage enabled.")

    
    # =====================================================
    # WRITABLE KB SHADOW MODE
    # =====================================================

    writable_kb = None

    if args.shadow_kb:
        writable_kb = WritableKB()

        print("Writable KB shadow mode enabled.")

        promoted = writable_kb.list_promoted()

        print(
            f"Writable KB promoted patterns: {len(promoted)}"
        )


    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid4().hex[:8]
    )

    run_output_dir = OUTPUT_DIR / run_id

    run_output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    print(f"Run ID: {run_id}")

    total = min(args.limit, len(data))

    print(f"Processing {total} network flows...")

    summary_rows = []

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

        
        # =====================================================
        # INCOMING HISTORY MODE
        # =====================================================

        if incoming_context is not None:

            probabilities = pipeline.model.predict_proba(
                flow[pipeline.feature_names]
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

                    earlier_batch = data.iloc[:index].copy()

                    (
                        prepared_ae_flow,
                        historical_flow_count
                    ) = incoming_context.prepare(
                        flow,
                        earlier_batch_flows=earlier_batch
                    )


        alert = pipeline.predict_one(
            flow=flow,
            alert_id=alert_id,
            prepared_ae_flow=prepared_ae_flow,
            historical_flow_count=historical_flow_count,
            asset_id=args.asset_id,
            acceptable_risk=args.acceptable_risk,
        )

        
        # =====================================================
        # HYBRID SHADOW SCREENING
        # =====================================================

        hybrid_result = None
        hybrid_diagnostics = None

        if args.hybrid_screening:

            hybrid_result = {
                "original_status": alert.detection_status,
                "shadow_status": "NOT_SCREENED",
                "decision_changed": False,
                "requires_review": False,
                "reason": "Screening applies to high-confidence BENIGN flows",
            }

            is_high_confidence_benign = (
                alert.detection_status == "BENIGN"
                and alert.evidence.get("confidence_route")
                == "HIGH_CONFIDENCE"
            )

            if is_high_confidence_benign:

                kb_result = explain_kb_match(
                    flow=flow,
                    artifact=pipeline.kb_artifact,
                    reference=pipeline.kb_reference,
                    top_n=5,
                )

                ae_result = None
                context_count = None

                if not kb_result["kb_matched"]:

                    if history is not None:
                        ae_flow, context_count = history.prepare(
                            args.known_test_row
                        )

                    elif incoming_context is not None:
                        earlier_batch = data.iloc[:index].copy()

                        ae_flow, context_count = (
                            incoming_context.prepare(
                                flow,
                                earlier_batch_flows=earlier_batch,
                            )
                        )

                    else:
                        ae_flow = None

                    if ae_flow is not None:

                        if shadow_ae is None:
                            shadow_ae = AutoencoderIntegration()

                        ae_result, _ = shadow_ae.analyze(
                            ae_flow
                        )

                hybrid_result = resolve_hybrid_decision(
                    production_alert=alert,
                    kb_result=kb_result,
                    ae_result=ae_result,
                )

                hybrid_diagnostics = {
                    "frozen_kb_independent_lookup": kb_result,
                    "ae_independent": ae_result,
                    "historical_flow_count": context_count,
                }

            print("\n========== HYBRID SHADOW DECISION ==========")
            print("Original:", hybrid_result["original_status"])
            print("Shadow:", hybrid_result["shadow_status"])
            print("Changed:", hybrid_result["decision_changed"])
            print("Review:", hybrid_result["requires_review"])
            print("Reason:", hybrid_result["reason"])
            print("============================================")

        
        # =====================================================
        # STORE POTENTIAL UNKNOWN KB CANDIDATES
        # =====================================================

        if candidate_store is not None:

            eligible = (
                alert.detection_status == "POTENTIAL_NEW_UNKNOWN"
                and alert.evidence.get("confidence_route")
                    == "LOW_CONFIDENCE"
                and alert.evidence.get("kb_called") is True
                and alert.evidence.get("kb_matched") is False
                and alert.evidence.get("ae_called") is True
                and alert.evidence.get("ae_anomalous") is True
                and alert.evidence.get("kb_updated") is False
            )

            if eligible:

                # Store the original flow, not its
                # XGBoost or Autoencoder transformed values.
                flow_record = flow.iloc[0].to_dict()

                # Remove ground-truth labels and dataset
                # annotations from the candidate record.
                for column in (
                    "Label",
                    "Original_Label",
                    "source_file",
                    "Attempted Category",
                    "is_attempted",
                ):
                    flow_record.pop(column, None)

                candidate_evidence = {
                    "detection_status": alert.detection_status,
                    "predicted_class": alert.predicted_class,
                    "detection_confidence": (
                        alert.detection_confidence
                    ),
                    "confidence_route": (
                        alert.evidence.get("confidence_route")
                    ),
                    "kb_matched": False,
                    "nearest_pattern_id": (
                        alert.evidence.get("nearest_pattern_id")
                    ),
                    "nearest_distance": (
                        alert.evidence.get("nearest_distance")
                    ),
                    "ae_anomalous": True,
                    "ae_mse": alert.evidence.get("ae_mse"),
                    "ae_threshold": (
                        alert.evidence.get("ae_threshold")
                    ),
                    "historical_comparison": (
                        alert.evidence.get(
                            "historical_comparison"
                        )
                    ),
                    "source_alert_id": alert.alert_id,
                    "source_run_id": run_id,
                }

                candidate = candidate_store.add(
                    flow=flow_record,
                    evidence=candidate_evidence,
                )

                alert.evidence["kb_candidate_id"] = (
                    candidate["candidate_id"]
                )

                alert.evidence["kb_candidate_status"] = (
                    candidate["status"]
                )

                alert.evidence["kb_candidate_created"] = (
                    candidate["created"]
                )

                print(
                    "\nKB candidate stored:"
                    f" {candidate['candidate_id']}"
                )

                print(
                    "Candidate status:"
                    f" {candidate['status']}"
                )

                print(
                    "New candidate:"
                    f" {candidate['created']}"
                )

        
        # =====================================================
        # WRITABLE KB SHADOW LOOKUP
        # =====================================================

        if writable_kb is not None:

            # Use original flow values and the frozen
            # KB's saved 73-feature preprocessing.
            flow_record = flow.iloc[0].to_dict()

            shadow_result = writable_kb.lookup(
                flow_record
            )

            # Shadow lookup is evidence only.
            # It must not change detection status.
            alert.evidence["writable_kb_shadow"] = (
                shadow_result
            )

            print("\n--- WRITABLE KB SHADOW ---")
            print("Matched:", shadow_result["matched"])
            print(
                "Nearest pattern:",
                shadow_result["nearest_pattern_id"]
            )
            print(
                "Distance:",
                shadow_result["nearest_distance"]
            )
            print("Mode:", shadow_result["mode"])


        summary_rows.append({
            "alert_id": alert.alert_id,
                        "shadow_status": (
                hybrid_result["shadow_status"]
                if hybrid_result is not None
                else None
            ),
            "decision_changed": (
                hybrid_result["decision_changed"]
                if hybrid_result is not None
                else None
            ),
            "requires_review": (
                hybrid_result["requires_review"]
                if hybrid_result is not None
                else None
            ),
            "detection_status": alert.detection_status,
            "predicted_class": alert.predicted_class,
            "confidence": alert.detection_confidence,
            "kb_matched": alert.evidence.get("kb_matched"),
            "ae_anomalous": alert.evidence.get("ae_anomalous"),
            "risk_score": (
                alert.risk.inherent_risk
                if alert.risk is not None
                else None
            ),
            "risk_decision": (
                alert.risk.decision
                if alert.risk is not None
                else "NOT_AVAILABLE"
            ),
            "nearest_historical_attack": (
                alert.evidence.get(
                    "historical_comparison", {}
                ).get("nearest_historical_attack")
            ),

            "historical_distance": (
                alert.evidence.get(
                    "historical_comparison", {}
                ).get("distance")
            ),
            "kb_candidate_id": alert.evidence.get(
                "kb_candidate_id"
            ),

            "kb_candidate_status": alert.evidence.get(
                "kb_candidate_status"
            ),
            
            "shadow_kb_matched": (
                alert.evidence.get(
                    "writable_kb_shadow", {}
                ).get("matched")
            ),

            "shadow_kb_pattern_id": (
                alert.evidence.get(
                    "writable_kb_shadow", {}
                ).get("nearest_pattern_id")
            ),

            "shadow_kb_distance": (
                alert.evidence.get(
                    "writable_kb_shadow", {}
                ).get("nearest_distance")
            ),
            "hybrid_kb_matched": (
                hybrid_diagnostics["frozen_kb_independent_lookup"]["kb_matched"]
                if hybrid_diagnostics is not None
                else None
            ),
            "hybrid_ae_anomalous": (
                hybrid_diagnostics["ae_independent"]["anomalous"]
                if hybrid_diagnostics is not None
                and hybrid_diagnostics["ae_independent"] is not None
                else None
            ),
            "hybrid_ae_mse": (
                hybrid_diagnostics["ae_independent"]["anomaly_mse"]
                if hybrid_diagnostics is not None
                and hybrid_diagnostics["ae_independent"] is not None
                else None
            ),
            "risk_level": (
                alert.risk.risk_level
                if alert.risk is not None
                else "NOT_ASSESSED"
            ),

        })


        print("\n" + format_alert(alert))

        output_path = (
            run_output_dir
            / f"{safe_filename(alert_id)}.json"
        )

        payload = alert_to_dict(alert)

        if args.hybrid_screening:
            payload["hybrid_shadow_decision"] = hybrid_result
            payload["hybrid_diagnostics"] = hybrid_diagnostics

        with output_path.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False
            )

        
        print(f"\nSaved JSON: {output_path}")

        # =====================================================
        # SAVE BATCH SUMMARY
        # =====================================================

    summary = pd.DataFrame(summary_rows)

    summary_path = run_output_dir / "summary.csv"

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\n========== RUN SUMMARY ==========")

    print(summary.to_string(index=False))

    print(f"\nSummary saved to: {summary_path}")

    print("\n========== COMPLETE ==========")


if __name__ == "__main__":
    main()
