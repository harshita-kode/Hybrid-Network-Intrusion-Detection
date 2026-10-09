"""Shared V7 preprocessing and scoring. Never refit during evaluation.

V7 = normal-only autoencoder. Nothing in this module (or in the training notebook) uses attack data:
preprocessing, port table, weights, early stopping and threshold are all fit on BENIGN rows only.
A flow whose reconstruction error exceeds the benign-validation threshold is labelled UNKNOWN,
otherwise NORMAL.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

VERSION = 'v7'
SPLIT_TAG = 'v5'          # hold-out file written by v5; used ONLY for reporting, never for fitting or selection
LABELS = ('NORMAL', 'UNKNOWN')

FEATURES = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Fwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']

PORT_COL = FEATURES.index('Destination Port')
RARITY_CAP = 4.5   # = frequency 3e-5; stops one rare benign port dominating the reconstruction error

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
    if benign_only:
        if 'Label' not in df:
            raise ValueError(f"{name}: benign-only data must carry a Label column to prove it is benign")
        if not df['Label'].astype(str).str.strip().str.upper().eq('BENIGN').all():
            raise ValueError(f"{name}: expected benign only")
    return df, x

def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

def fit_port_table(ports, min_count=15):
    """-log10 frequency of each destination port in BENIGN TRAINING data only."""
    ports = ports.astype(np.int64)
    values, counts = np.unique(ports, return_counts=True)
    keep = counts >= min_count
    table = {int(v): float(-np.log10(c / len(ports))) for v, c in zip(values[keep], counts[keep])}
    unseen = float(-np.log10(1.0 / (len(ports) + 1)))
    return table, unseen

def engineer(x, prep):
    """[port_rarity, other 68 features, port<1024, 1024-49151, >=49152]. Raw port number is removed."""
    ports = x[:, PORT_COL]
    rarity = pd.Series(ports.astype(np.int64)).map(prep['port_table']).fillna(prep['port_unseen']).to_numpy(np.float64)
    rarity = np.minimum(rarity, RARITY_CAP)
    flags = np.column_stack([ports < 1024, (ports >= 1024) & (ports < 49152), ports >= 49152]).astype(np.float64)
    return np.column_stack([rarity, np.delete(x, PORT_COL, axis=1), flags])

def fit_preprocessing(train):
    prep = {'features': FEATURES.copy(), 'transform': 'signed_log1p_v1', 'port_mode': 'engineered'}
    prep['port_table'], prep['port_unseen'] = fit_port_table(train[:, PORT_COL])
    t = engineer(train, prep)
    q50 = np.median(np.abs(t), axis=0)
    q99 = np.percentile(np.abs(t), 99, axis=0)
    mask = (q99 > 100 * np.maximum(q50, 1)) & (q99 > 100)   # skew decision from benign training rows only
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

def reconstruction_mse(models, x, batch_size=4096):
    """Per-flow reconstruction MSE, averaged over the seed ensemble."""
    return np.mean([np.mean((x - m.predict(x, batch_size=batch_size, verbose=0)) ** 2, axis=1) for m in models], axis=0)

def benign_threshold(scores, fpr):
    return float(np.percentile(scores, 100 * (1 - fpr)))

def classify(scores, threshold):
    """NORMAL if reconstruction error <= threshold, else UNKNOWN."""
    return np.where(np.asarray(scores) > threshold, LABELS[1], LABELS[0])
