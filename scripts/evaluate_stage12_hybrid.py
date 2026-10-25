"""GC2 Stage 12E: combine saved KB/AE results with XGBoost routing.

Read-only offline evaluation. It does NOT call GC2Pipeline.predict_one and
must not be described as an end-to-end runtime benchmark.
"""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline.integration import GC2Pipeline

DATA = ROOT / 'data/processed/Data_V3_SSH_Unknown_final'
OUT = ROOT / 'outputs/results/gc2_evaluation/stage12'
THRESHOLD = 0.95


def main():
    started = time.perf_counter()
    kb = pd.read_csv(OUT / 'kb_retrieval_predictions.csv')
    ae = pd.read_csv(OUT / 'ae_predictions.csv')
    unknown = pd.read_parquet(DATA / 'unknown_kb_splits/unknown_final_test.parquet')
    known = pd.read_parquet(DATA / 'known_test.parquet')
    benign = known.loc[known['Label'].eq('BENIGN')].sample(n=len(unknown), random_state=42)
    flows = pd.concat([unknown, benign], ignore_index=True)
    if len(kb) != len(flows) or len(flows) != 2602:
        raise ValueError('Stage 12B and source dataset counts do not agree')
    if not np.array_equal(flows['Label'].astype(str).to_numpy(), kb['actual_label'].astype(str).to_numpy()):
        raise ValueError('Stage 12B rows are not aligned with the input flows')
    if kb['kb_matched'].isna().any():
        raise ValueError('Missing KB decisions')
    matched = kb['kb_matched'].astype(bool).to_numpy()
    miss_idx = np.flatnonzero(~matched)
    if len(ae) != len(miss_idx):
        raise ValueError('Stage 12D row count differs from KB miss count')
    if not np.array_equal(flows.iloc[miss_idx]['Label'].astype(str).to_numpy(), ae['actual_label'].astype(str).to_numpy()):
        raise ValueError('Stage 12D AE rows are not in the expected KB-miss order')
    if ae['anomalous'].isna().any():
        raise ValueError('Missing AE decisions')
    if not np.all(np.isfinite(ae['anomaly_mse'].to_numpy(dtype=float))):
        raise ValueError('Nonfinite AE MSE')
    if not np.array_equal(ae['anomalous'].astype(bool).to_numpy(),
                          (ae['anomaly_mse'].to_numpy(dtype=float) > ae['threshold'].to_numpy(dtype=float))):
        raise ValueError('AE anomaly decisions disagree with saved thresholds')

    pipeline = GC2Pipeline()
    # The model's numeric class mapping must be verified against the training encoder.
    classes = sorted(pd.read_parquet(DATA / 'known_train.parquet', columns=['Label'])['Label'].unique().tolist())
    if len(classes) != len(pipeline.model.classes_):
        raise ValueError('Class count mismatch; verify original training encoder')
    class_ids = np.asarray(pipeline.model.classes_, dtype=int)
    if not np.array_equal(class_ids, np.arange(len(classes))):
        raise ValueError('Unexpected model class IDs; verify training encoder')
    missing = [c for c in pipeline.feature_names if c not in flows.columns]
    if missing:
        raise ValueError(f'Missing XGBoost inputs: {missing}')

    predictions = []
    confidences = []
    for begin in range(0, len(flows), 10000):
        proba = pipeline.model.predict_proba(flows.iloc[begin:begin+10000][pipeline.feature_names])
        idx = proba.argmax(axis=1)
        predictions.extend([classes[class_ids[i]] for i in idx])
        confidences.extend(proba.max(axis=1).tolist())

    result = pd.DataFrame({
        'actual_label': flows['Label'].astype(str).to_numpy(),
        'group': kb['group'].to_numpy(),
        'xgb_prediction': predictions,
        'xgb_confidence': confidences,
        'kb_matched': matched,
        'kb_distance': kb['nearest_distance'].to_numpy(),
        'ae_anomalous': pd.Series([pd.NA] * len(flows), dtype='boolean'),
        'ae_mse': np.full(len(flows), np.nan),
    })
    result.loc[miss_idx, 'ae_anomalous'] = ae['anomalous'].astype(bool).to_numpy()
    result.loc[miss_idx, 'ae_mse'] = ae['anomaly_mse'].to_numpy()
    high = result['xgb_confidence'].ge(THRESHOLD)
    benign_pred = result['xgb_prediction'].eq('BENIGN')
    known_pred = ~benign_pred
    # This is a routing approximation, not the exact production alert state.
    original = np.select(
        [high & known_pred, high & benign_pred, ~high & matched,
         ~high & ~matched & result['ae_anomalous'].fillna(False).astype(bool)],
        ['KNOWN_ATTACK', 'BENIGN', 'UNCERTAIN_KB_MATCH', 'POTENTIAL_NEW_UNKNOWN'],
        default='UNCERTAIN',
    )
    result['original_route_approx'] = original
    shadow = np.array(original, dtype=object)
    screened = high & benign_pred
    shadow[screened & matched] = 'UNCERTAIN_KB_MATCH'
    shadow[screened & ~matched & result['ae_anomalous'].fillna(False).astype(bool)] = 'POTENTIAL_NEW_UNKNOWN'
    result['shadow_route_approx'] = shadow
    result['shadow_changed'] = result['original_route_approx'].ne(result['shadow_route_approx'])
    result['kb_or_ae_signal'] = matched | result['ae_anomalous'].fillna(False).astype(bool).to_numpy()
    result['shadow_review_flag'] = result['shadow_route_approx'].isin(['UNCERTAIN_KB_MATCH', 'POTENTIAL_NEW_UNKNOWN', 'UNCERTAIN'])
    result['original_review_flag'] = result['original_route_approx'].isin(['UNCERTAIN_KB_MATCH', 'POTENTIAL_NEW_UNKNOWN', 'UNCERTAIN'])
    result.to_csv(OUT / 'hybrid_predictions.csv', index=False)

    def stats(sub):
        return {
            'samples': int(len(sub)),
            'high_confidence_benign': int((sub['xgb_prediction'].eq('BENIGN') & sub['xgb_confidence'].ge(THRESHOLD)).sum()),
            'kb_matches': int(sub['kb_matched'].sum()),
            'ae_anomalies_among_kb_misses': int(sub['ae_anomalous'].fillna(False).astype(bool).sum()),
            'kb_or_ae_signals': int(sub['kb_or_ae_signal'].sum()),
            'original_review_flags_approx': int(sub['original_review_flag'].sum()),
            'shadow_review_flags_approx': int(sub['shadow_review_flag'].sum()),
            'new_review_flags_approx': int((sub['shadow_review_flag'] & ~sub['original_review_flag']).sum()),
        }

    unknown_rows = result[result['group'].eq('UNKNOWN')]
    benign_rows = result[result['group'].eq('BENIGN')]
    metrics = {
        'unknown': stats(unknown_rows), 'benign': stats(benign_rows),
        'threshold': THRESHOLD,
        'elapsed_seconds': time.perf_counter() - started,
        'warnings': [
            'Offline approximation: not exact GC2Pipeline.predict_one decisions.',
            'KB matches are retrieval signals, not confirmed malicious detections.',
            'Known-class XGBoost numeric mapping inferred from sorted training labels; verify training encoder.',
            'XGBoost includes potential leakage features id, Attempted Category, is_attempted.',
            'Random split may share capture-session patterns; AE training/calibration independence unverified.',
            'This is not a production throughput benchmark.',
        ],
    }
    (OUT / 'hybrid_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    per_class = result.groupby('actual_label', as_index=False).agg(
        samples=('actual_label', 'size'),
        xgb_high_conf_benign=('xgb_confidence', lambda s: int(((s >= THRESHOLD) & result.loc[s.index, 'xgb_prediction'].eq('BENIGN')).sum())),
        kb_matches=('kb_matched', 'sum'),
        kb_or_ae_signals=('kb_or_ae_signal', 'sum'),
        original_review_flags=('original_review_flag', 'sum'),
        shadow_review_flags=('shadow_review_flag', 'sum'),
    )
    per_class.to_csv(OUT / 'hybrid_per_class.csv', index=False)
    print('\n========== STAGE 12E OFFLINE HYBRID COMPARISON ==========')
    print('UNKNOWN:', metrics['unknown'])
    print('BENIGN:', metrics['benign'])
    print('\n', per_class.to_string(index=False))
    print('\nSaved hybrid_predictions.csv, hybrid_metrics.json, hybrid_per_class.csv')
    print('STAGE 12E COMPLETE (PROVISIONAL)')


if __name__ == '__main__':
    main()
