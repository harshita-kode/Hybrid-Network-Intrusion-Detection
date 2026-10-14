"""Read-only GC2 demo on existing KB demo Parquet files.

Production decision and independent diagnostic evidence are deliberately separate.
Run from repository root: python -m scripts.run_gc2_demo --scenario matched --row 0
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.pipeline.integration import GC2Pipeline
from src.pipeline.alert import alert_to_dict, format_alert
from src.analysis.kb_explanation import explain_kb_match

ROOT = Path(__file__).resolve().parents[1]
# Allow joblib to resolve the original module name
# stored inside the supervised KNN artifact.
KNN_KNOWN_DIR = ROOT / "notebooks" / "knn_known"

if not KNN_KNOWN_DIR.is_dir():
    raise FileNotFoundError(
        f"Known-attack KNN module directory missing: {KNN_KNOWN_DIR}"
    )

if str(KNN_KNOWN_DIR) not in sys.path:
    sys.path.insert(0, str(KNN_KNOWN_DIR))

DEMO_DIR = ROOT / 'data/processed/knn_kb/demo_similarity_audit'
SCENARIOS = {
    'matched': DEMO_DIR / 'kb_demo_matched.parquet',
    'unmatched': DEMO_DIR / 'autoencoder_demo_unmatched.parquet',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=SCENARIOS, required=True)
    parser.add_argument('--row', type=int, default=0, help='Positional row within demo Parquet')
    parser.add_argument('--history', type=Path, help='Earlier chronological flow history, CSV or Parquet')
    parser.add_argument('--asset-id', default=None)
    parser.add_argument('--acceptable-risk', type=float, default=None)
    args = parser.parse_args()

    path = SCENARIOS[args.scenario]
    if not path.is_file():
        parser.error(f'Missing demo data: {path}')
    data = pd.read_parquet(path)
    if not 0 <= args.row < len(data):
        parser.error(f'--row must be between 0 and {len(data)-1}')
   
    # Load the selected label-free demo record.
    demo_flow = data.iloc[[args.row]].copy()

    if "demo_row" not in demo_flow.columns:
        raise ValueError(
            "Demo file is missing the original demo_row identifier"
        )

    demo_position = int(demo_flow.iloc[0]["demo_row"])

    # Recover the original record from the saved demo partition.
    source_path = (
        ROOT
        / "data/processed/Data_V3_SSH_Unknown_final"
        / "unknown_kb_splits"
        / "unknown_demo_kb_updates.parquet"
    )

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Original demo source missing: {source_path}"
        )

    source = pd.read_parquet(source_path)

    if not 0 <= demo_position < len(source):
        raise IndexError("Invalid original demo row")

    flow = source.iloc[[demo_position]].copy()

    # Check that the restored record really is the selected
    # demo record before passing it to XGBoost.
    common_columns = [
        c for c in demo_flow.columns
        if c != "demo_row" and c in flow.columns
    ]

    if not common_columns:
        raise ValueError("No common columns for demo verification")

    import pandas.testing as pdt

    pdt.assert_frame_equal(
        demo_flow[common_columns].reset_index(drop=True),
        flow[common_columns].reset_index(drop=True),
        check_dtype=False,
    )

    pipeline = GC2Pipeline()

    missing = [
        name for name in pipeline.feature_names
        if name not in flow.columns
    ]

    if missing:
        raise ValueError(
            f"Original demo source still lacks XGBoost features: {missing}"
        )

    print("Original demo row:", demo_position)
    print("XGBoost feature restoration: PASSED")

    # Always perform a separate KB diagnostic, even if XGBoost is confident.
    kb = explain_kb_match(flow, pipeline.kb_artifact, pipeline.kb_reference)
    prepared = None
    historical_count = None
    if args.history is not None:
        from src.pipeline.incoming_context import IncomingContext
        if not args.history.is_file():
            parser.error(f'History file not found: {args.history}')
        if args.history.suffix.lower() == '.parquet':
            history = pd.read_parquet(args.history)
        elif args.history.suffix.lower() == '.csv':
            history = pd.read_csv(args.history)
        else:
            parser.error('History must be CSV or Parquet')
        # IncomingContext selects strictly earlier timestamps; user must supply
        # a valid, non-overlapping history with appropriate arrival ordering.
        prepared, historical_count = IncomingContext(history).prepare(flow)

    alert = pipeline.predict_one(
        flow=flow,
        alert_id='GC2-DEMO-001',
        prepared_ae_flow=prepared,
        historical_flow_count=historical_count,
        asset_id=args.asset_id,
        acceptable_risk=args.acceptable_risk,
    )

    diagnostics = {'frozen_kb_independent_lookup': kb,
                   'ae_independent': None,
                   'known_attack_knn_comparison': None}
    if not kb['kb_matched'] and prepared is not None:
        from src.pipeline.ae_integration import AutoencoderIntegration
        ae = pipeline.ae_integration or AutoencoderIntegration()
        result, _ = ae.analyze(prepared)
        diagnostics['ae_independent'] = result
    elif not kb['kb_matched']:
        diagnostics['ae_independent'] = {'not_run': 'No --history provided'}

    # Known-attack KNN is a *comparison*, not a final detection override.
    if args.scenario == 'unmatched':
        from notebooks.knn_known.knn_known_flow import load_artifact, predict_rows
        artifact_path = ROOT / 'saved-models/knn_known/known_classifier.joblib'
        if artifact_path.is_file():
            known = load_artifact(artifact_path)
            comparison = {k: (v.item() if hasattr(v, 'item') else v) for k, v in predict_rows(flow, known).iloc[0].to_dict().items()}
            diagnostics['known_attack_knn_comparison'] = comparison
        else:
            diagnostics['known_attack_knn_comparison'] = {'not_run': 'Model missing'}

    from src.pipeline.hybrid_decision import (
    resolve_hybrid_decision
    )

    hybrid_decision = resolve_hybrid_decision(
        production_alert=alert,
        kb_result=diagnostics["frozen_kb_independent_lookup"],
        ae_result=diagnostics["ae_independent"],
    )

    payload = {
        'source_demo_file': str(path.relative_to(ROOT)),
        'source_demo_row': args.row,
        'production_routing_alert': alert_to_dict(alert),
        'independent_diagnostics': diagnostics,
        "hybrid_shadow_decision": hybrid_decision,
        'notes': [
            'Independent diagnostics never override the production alert.',
            'Frozen KB matches are retrieval evidence, not attack proof.',
            'Known-class KNN vote share is not a calibrated probability.',
            'No KB updates or candidate promotion were performed.',
        ],
    }
    output_dir = ROOT / 'outputs/alerts' / ('demo_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8])
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / 'demo_result.json'
    with output.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(format_alert(alert))
    print('\nIndependent frozen KB match:', kb['kb_matched'])
    print('Independent AE run:', diagnostics['ae_independent'] is not None and 'not_run' not in diagnostics['ae_independent'])
    print('Known-attack KNN comparison:', diagnostics['known_attack_knn_comparison'])
    print("\n========== HYBRID SHADOW DECISION ==========")
    print("Original status:", hybrid_decision["original_status"])
    print("Shadow status:", hybrid_decision["shadow_status"])
    print("Changed:", hybrid_decision["decision_changed"])
    print("Requires review:", hybrid_decision["requires_review"])
    print("Reason:", hybrid_decision["reason"])
    print("===============================================")
    print('Saved:', output)


if __name__ == '__main__':
    main()
