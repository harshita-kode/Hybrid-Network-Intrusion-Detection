"""GC2 Stage 12F: audit saved offline routing results without rerunning models.

Checks Stage 12E routing algebra against the stated shadow policy.
This is NOT an end-to-end verification of GC2Pipeline.predict_one.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/results/gc2_evaluation/stage12'
THRESHOLD = 0.95


def main():
    path = OUT / 'hybrid_predictions.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = ['actual_label', 'group', 'xgb_prediction', 'xgb_confidence',
                'kb_matched', 'ae_anomalous', 'original_route_approx',
                'shadow_route_approx', 'original_review_flag',
                'shadow_review_flag', 'shadow_changed']
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f'Missing Stage 12E columns: {missing}')
    if df[required[:5] + required[6:]].isna().any().any():
        raise ValueError('Unexpected missing required values')
    if df['xgb_confidence'].lt(0).any() or df['xgb_confidence'].gt(1).any():
        raise ValueError('Invalid XGBoost confidence')

    kb = df['kb_matched'].astype(bool).to_numpy()
    ae = df['ae_anomalous'].astype('boolean').fillna(False).to_numpy(dtype=bool)
    high = df['xgb_confidence'].to_numpy() >= THRESHOLD
    benign = df['xgb_prediction'].eq('BENIGN').to_numpy()
    screened = high & benign

    # Mirrors the original offline routing approximation in Stage 12E.
    original = np.select(
        [high & ~benign, screened, ~high & kb, ~high & ~kb & ae],
        ['KNOWN_ATTACK', 'BENIGN', 'UNCERTAIN_KB_MATCH', 'POTENTIAL_NEW_UNKNOWN'],
        default='UNCERTAIN',
    ).astype(object)
    shadow = original.copy()
    shadow[screened & kb] = 'UNCERTAIN_KB_MATCH'
    shadow[screened & ~kb & ae] = 'POTENTIAL_NEW_UNKNOWN'
    # High-confidence benign + KB miss + AE negative stays BENIGN.
    review_states = {'UNCERTAIN_KB_MATCH', 'POTENTIAL_NEW_UNKNOWN', 'UNCERTAIN'}
    orig_review = np.isin(original, list(review_states))
    shadow_review = np.isin(shadow, list(review_states))
    changed = original != shadow

    checks = {
        'original_route': np.array_equal(original, df['original_route_approx'].to_numpy()),
        'shadow_route': np.array_equal(shadow, df['shadow_route_approx'].to_numpy()),
        'original_review': np.array_equal(orig_review, df['original_review_flag'].astype(bool).to_numpy()),
        'shadow_review': np.array_equal(shadow_review, df['shadow_review_flag'].astype(bool).to_numpy()),
        'changed': np.array_equal(changed, df['shadow_changed'].astype(bool).to_numpy()),
        'known_attack_preserved': bool(np.all(shadow[high & ~benign] == 'KNOWN_ATTACK')),
        'unscreened_unchanged': bool(np.all(~changed[~screened])),
        'screened_kb_matches_reviewed': bool(np.all(shadow[screened & kb] == 'UNCERTAIN_KB_MATCH')),
        'screened_ae_anomalies_reviewed': bool(np.all(shadow[screened & ~kb & ae] == 'POTENTIAL_NEW_UNKNOWN')),
    }
    if not all(checks.values()):
        print('FAILED CHECKS:', [k for k, v in checks.items() if not v])
        raise AssertionError('Stage 12E routing inconsistency')

    summary = {}
    for group in ('UNKNOWN', 'BENIGN'):
        mask = df['group'].eq(group).to_numpy()
        summary[group] = {
            'samples': int(mask.sum()),
            'high_confidence_benign_screened': int((mask & screened).sum()),
            'original_review_flags': int((mask & orig_review).sum()),
            'shadow_review_flags': int((mask & shadow_review).sum()),
            'new_review_flags': int((mask & shadow_review & ~orig_review).sum()),
            'screened_kb_matches': int((mask & screened & kb).sum()),
            'screened_ae_anomalies_after_kb_miss': int((mask & screened & ~kb & ae).sum()),
        }
    report = {
        'result': 'PASSED', 'rows': int(len(df)),
        'checks': checks, 'groups': summary,
        'scope': 'Saved offline routing algebra only; not runtime predict_one equivalence.',
        'limitations': [
            'Unverified KB pattern provenance is treated as review evidence, not confirmed malicious detection.',
            'XGBoost class mapping and label/identifier leakage remain unresolved.',
            'AE training/calibration independence and same-session correlation remain unverified.',
        ],
    }
    output = OUT / 'routing_consistency_audit.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('========== STAGE 12F ROUTING AUDIT ==========')
    print('Rows:', len(df))
    for group, stats in summary.items():
        print(group, stats)
    print('All offline routing checks: PASSED')
    print('Saved:', output)
    print('NOTE: This does not validate production pipeline decisions.')


if __name__ == '__main__':
    main()
