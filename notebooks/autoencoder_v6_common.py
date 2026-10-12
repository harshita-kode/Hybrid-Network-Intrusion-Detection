"""Shared V6 preprocessing and scoring. Never refit during evaluation.

Changes vs V5
-------------
* Destination Port encoding ('engineered'): raw port number is replaced by its RARITY in benign
  training traffic plus well-known / registered / dynamic flags. Raw ports 22, 53, 80, 443 are
  numerically almost identical after standardising, so scans on rare ports were invisible.
* Isolation-Forest score and tail-probability (Fisher-style) score fusion with the autoencoder.
* Seed-ensemble scoring helper shared by training and testing notebooks.
* Pruned scores that never won in V5 (latent_mahal, normalized_mae).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

VERSION = 'v6'
SPLIT_TAG = 'v5'   # calibration / hold-out split files are reused from V5 so results are comparable

FEATURES = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Fwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']

PORT_COL = FEATURES.index('Destination Port')
RARITY_CAP = 4.5   # = frequency 3e-5; stops one rare benign port dominating the reconstruction error
BASE_SCORES = ['mse', 'mae', 'max_abs', 'p95_mean', 'p95_top5']
FUSIONS = {'fuse_mse_iforest': ('mse', 'iforest'),
           'fuse_p95top5_iforest': ('p95_top5', 'iforest'),
           'fuse_mse_p95top5': ('mse', 'p95_top5')}

def validate_frame(df, name, benign_only=False):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    if df.columns.duplicated().any():
        raise ValueError(f"{name}: duplicated column names")
    missing = [x for x in FEATURES if x not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing features: {missing}")
    for feature in FEATURES:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise ValueError(f"{name}: nonnumeric {feature}")
    x = df[FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f"{name}: NaN or infinite feature values")
    if benign_only and 'Label' in df:
        if not df['Label'].astype(str).str.strip().str.upper().eq('BENIGN').all():
            raise ValueError(f"{name}: expected benign only")
    return df, x

def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

# ---------------------------------------------------------------- port engineering
def fit_port_table(ports, min_count=15):
    """-log10 frequency of each destination port in BENIGN TRAINING data only."""
    ports = ports.astype(np.int64)
    values, counts = np.unique(ports, return_counts=True)
    keep = counts >= min_count
    table = {int(v): float(-np.log10(c / len(ports))) for v, c in zip(values[keep], counts[keep])}
    unseen = float(-np.log10(1.0 / (len(ports) + 1)))   # ports never / almost never seen in benign traffic
    return table, unseen

def engineer(x, prep):
    """raw: unchanged. engineered: [port_rarity, other 68 features, port<1024, 1024-49151, >=49152]."""
    if prep.get('port_mode', 'raw') != 'engineered':
        return x
    ports = x[:, PORT_COL]
    rarity = pd.Series(ports.astype(np.int64)).map(prep['port_table']).fillna(prep['port_unseen']).to_numpy(np.float64)
    rarity = np.minimum(rarity, RARITY_CAP)
    flags = np.column_stack([ports < 1024, (ports >= 1024) & (ports < 49152), ports >= 49152]).astype(np.float64)
    return np.column_stack([rarity, np.delete(x, PORT_COL, axis=1), flags])

def fit_preprocessing(train, mode='signedlog_standard', port_mode='engineered'):
    if mode not in ('standard', 'signedlog_standard'):
        raise ValueError(mode)
    if port_mode not in ('raw', 'engineered'):
        raise ValueError(port_mode)
    prep = {'mode': mode, 'port_mode': port_mode, 'features': FEATURES.copy(), 'transform': 'signed_log1p_v1'}
    if port_mode == 'engineered':
        prep['port_table'], prep['port_unseen'] = fit_port_table(train[:, PORT_COL])
    t = engineer(train, prep)
    # Skew decision learned only from benign training rows.
    q50 = np.median(np.abs(t), axis=0)
    q99 = np.percentile(np.abs(t), 99, axis=0)
    mask = (q99 > 100 * np.maximum(q50, 1)) & (q99 > 100)
    if mode == 'standard':
        mask[:] = False
    t = t.copy()
    t[:, mask] = signed_log(t[:, mask])
    scaler = StandardScaler().fit(t)
    prep.update({'log_mask': mask.astype(bool).tolist(), 'scaler': scaler, 'input_dim': int(t.shape[1])})
    return prep

def transform(x, preprocessor):
    if preprocessor['features'] != FEATURES:
        raise ValueError('Feature order mismatch')
    z = engineer(x, preprocessor).copy()
    mask = np.asarray(preprocessor['log_mask'], dtype=bool)
    z[:, mask] = signed_log(z[:, mask])
    z = preprocessor['scaler'].transform(z).astype(np.float32)
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite values after preprocessing')
    return z

# ---------------------------------------------------------------- scoring
def residual_scales(x, reconstructed):
    """Per-feature 95th-percentile benign residual. Fit on benign VALIDATION data only."""
    return np.maximum(np.percentile(np.abs(x - reconstructed), 95, axis=0), 0.05).astype(np.float32)

def scores_by_method(x, reconstructed, p95_scale, top_k=5):
    res = np.abs(x - reconstructed).astype(np.float32)
    norm = res / p95_scale
    k = min(top_k, norm.shape[1])
    return {'mse': np.mean(res ** 2, axis=1), 'mae': np.mean(res, axis=1), 'max_abs': np.max(res, axis=1),
            'p95_mean': np.mean(norm, axis=1),
            'p95_top5': np.mean(np.partition(norm, -k, axis=1)[:, -k:], axis=1)}

def ensemble_base_scores(models, scales_list, x, batch_size=4096):
    """Average each autoencoder score over a list of (model, p95 scale) pairs."""
    per = [scores_by_method(x, m.predict(x, batch_size=batch_size, verbose=0), sc)
           for m, sc in zip(models, scales_list)]
    return {k: np.mean([p[k] for p in per], axis=0) for k in per[0]}

def tail_score(s, ref_sorted):
    """-log10 of the benign exceedance probability of s (ref_sorted = sorted benign validation scores)."""
    n = len(ref_sorted)
    greater = n - np.searchsorted(ref_sorted, s, side='right')
    return -np.log10((greater + 1.0) / (n + 1.0))

def all_scores(base, iforest_scores, refs=None):
    """Adds 'iforest' and fused scores. refs=None on benign validation (fits them); pass them for other sets."""
    s = dict(base)
    s['iforest'] = np.asarray(iforest_scores, dtype=np.float64)
    if refs is None:
        refs = {n: np.sort(s[n]) for n in {m for pair in FUSIONS.values() for m in pair}}
    for name, (a, b) in FUSIONS.items():
        s[name] = tail_score(s[a], refs[a]) + tail_score(s[b], refs[b])
    return s, refs

def benign_threshold(scores, fpr):
    return float(np.percentile(scores, 100 * (1 - fpr)))

def tpr_at_fpr(y_true, scores, fprs):
    fpr, tpr, _ = roc_curve(y_true, scores)
    return np.interp(np.asarray(fprs, dtype=float), fpr, tpr)
