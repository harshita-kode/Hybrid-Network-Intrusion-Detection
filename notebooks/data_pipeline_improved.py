"""GC2 Member 1: shared Improved CICIDS2017 preparation. Run from anywhere:
python prepare_gc2_improved_v3.py --input data/raw/improved_CIC-IDS2017
Dependencies: pandas numpy scikit-learn pyarrow

Creates new Data_V3; never overwrites Data_V2. Does not train models.
IMPORTANT: A new split requires retraining V15, unless its fitted data match exactly.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
UNKNOWN = ('Botnet', 'FTP-Patator', 'DoS Slowhttptest')

FEATURES = """Protocol,Flow Duration,Total Fwd Packet,Total Bwd packets,Total Length of Fwd Packet,Total Length of Bwd Packet,Fwd Packet Length Max,Fwd Packet Length Min,Fwd Packet Length Mean,Fwd Packet Length Std,Bwd Packet Length Max,Bwd Packet Length Min,Bwd Packet Length Mean,Bwd Packet Length Std,Flow Bytes/s,Flow Packets/s,Flow IAT Mean,Flow IAT Std,Flow IAT Max,Flow IAT Min,Fwd IAT Total,Fwd IAT Mean,Fwd IAT Std,Fwd IAT Max,Fwd IAT Min,Bwd IAT Total,Bwd IAT Mean,Bwd IAT Std,Bwd IAT Max,Bwd IAT Min,Fwd PSH Flags,Fwd URG Flags,Fwd RST Flags,Bwd RST Flags,Fwd Header Length,Bwd Header Length,Fwd Packets/s,Bwd Packets/s,Packet Length Min,Packet Length Max,Packet Length Mean,Packet Length Std,Packet Length Variance,FIN Flag Count,SYN Flag Count,RST Flag Count,PSH Flag Count,ACK Flag Count,URG Flag Count,CWR Flag Count,ECE Flag Count,Down/Up Ratio,Average Packet Size,Fwd Segment Size Avg,Bwd Segment Size Avg,Subflow Fwd Packets,Subflow Fwd Bytes,Subflow Bwd Packets,Subflow Bwd Bytes,FWD Init Win Bytes,Bwd Init Win Bytes,Fwd Act Data Pkts,Fwd Seg Size Min,Active Mean,Active Std,Active Max,Active Min,Idle Mean,Idle Std,Idle Max,Idle Min,ICMP Code,ICMP Type,Total TCP Flow Time""".split(',')


def canonical_label(label):
    s = re.sub(r'\s+', ' ', str(label).strip())
    s = re.sub(r'^Web Attack\s*[-–—�]\s*', 'Web Attack - ', s, flags=re.I)
    attempted = bool(re.search(r'\s*-\s*Attempted$', s, flags=re.I))
    s = re.sub(r'\s*-\s*Attempted$', '', s, flags=re.I).strip()
    alias = {'bot': 'Botnet', 'botnet': 'Botnet', 'ftp-bruteforce': 'FTP-Patator',
             'ftp brute force': 'FTP-Patator', 'ftp patator': 'FTP-Patator',
             'dos slowhttp test': 'DoS Slowhttptest', 'dos slowhttptest': 'DoS Slowhttptest',
             'dos slow http test': 'DoS Slowhttptest'}
    s = alias.get(s.casefold(), s)
    return s, attempted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/raw/improved_CIC-IDS2017',
                    help='Folder containing extracted Improved CICIDS2017 CSV files')
    ap.add_argument('--out', default='data/processed/Data_V3')
    args = ap.parse_args()
    source, out = Path(args.input).resolve(), Path(args.out).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f'Extracted dataset folder not found: {source}')
    names = sorted(p for p in source.rglob('*.csv')
                   if not p.name.startswith('._') and '__MACOSX' not in p.parts)
    if not names:
        raise ValueError(f'No CSV files found inside {source}')
    out.mkdir(parents=True, exist_ok=True)
    print('INPUT FOLDER:', source, '\nOUTPUT:', out, flush=True)
    print('CSV FILES:', len(names), flush=True)
    frames = []
    for path in names:
        print('Loading:', path.relative_to(source), flush=True)
        part = pd.read_csv(path, low_memory=False)
        part.columns = part.columns.astype(str).str.strip()
        if part.columns.duplicated().any():
            raise ValueError(f'Duplicate headers: {path}')
        missing = set(FEATURES + ['Label']) - set(part.columns)
        if missing:
            raise ValueError(f'{path.name}: missing required V15 columns: {sorted(missing)}')
        part['source_file'] = path.stem
        frames.append(part)
        print(f'Loaded {path.name}: {len(part):,} rows', flush=True)
    df = pd.concat(frames, ignore_index=True)
    del frames
    initial = len(df)
    print('MERGED:', df.shape, flush=True)
    canonical = df['Label'].map(canonical_label)
    df['Label'] = canonical.map(lambda p: p[0])
    attempted = canonical.map(lambda p: p[1]).to_numpy(dtype=bool)
    if 'Attempted Category' in df.columns:
        ac = pd.to_numeric(df['Attempted Category'], errors='coerce')
        attempted |= (ac.notna() & ac.ne(-1)).to_numpy()
    df['is_attempted'] = attempted
    print('Original family counts (before attempted policy):')
    print(df['Label'].value_counts().to_string(), flush=True)
    print('Attempted rows:', int(attempted.sum()), flush=True)
    # Audit attempted family before changing labels.
    audit = pd.crosstab(df['Label'], df['is_attempted'])
    audit.to_csv(out / 'attempted_audit.csv')
    # Prevent unknown-family attempted traffic being relabeled BENIGN and used for training.
    unknown_family_attempted = df['Label'].isin(UNKNOWN) & df['is_attempted']
    excluded_unknown_attempted = int(unknown_family_attempted.sum())
    df = df.loc[~unknown_family_attempted].copy()
    df.loc[df['is_attempted'], 'Label'] = 'BENIGN'
    # Keep port for metadata and V15 port thresholds; do not include it in model features.
    # No identifier, label, attempted marker or source file enters the model.
    required = FEATURES + ['Label', 'source_file']
    if 'Dst Port' in df.columns: required.append('Dst Port')
    df = df[required].copy()
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    vals = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    bad = vals.isna().any(axis=1)
    removed_nonfinite = int(bad.sum())
    df = df.loc[~bad].copy()
    # V15-aligned invalid-negative checks, but preserve small IAT negatives and -1 TCP window sentinels.
    negative_cols = [c for c in ['Flow Duration', 'Flow Bytes/s', 'Flow Packets/s',
                                 'Fwd Header Length', 'Bwd Header Length', 'Fwd Seg Size Min'] if c in df]
    invalid_negative = (df[negative_cols] < 0).any(axis=1)
    removed_negative = int(invalid_negative.sum())
    df = df.loc[~invalid_negative].copy()

    identity = FEATURES + (['Dst Port'] if 'Dst Port' in df else [])
    h = pd.util.hash_pandas_object(df[identity], index=False)
    label_counts = df.groupby(h, sort=False)['Label'].nunique()
    bad_hashes = label_counts.index[label_counts > 1]
    conflict_rows = int(h.isin(bad_hashes).sum())
    df = df.loc[~h.isin(bad_hashes)].copy()
    before_dup = len(df)
    df = df.drop_duplicates(subset=identity).reset_index(drop=True)
    duplicates = before_dup - len(df)
    print('Removed nonfinite:', removed_nonfinite, 'invalid negatives:', removed_negative,
          'conflicting rows:', conflict_rows, 'duplicates:', duplicates, flush=True)
    print('Excluded attempted held-out-family rows:', excluded_unknown_attempted, flush=True)
    print('CLEAN:', df.shape, '\n', df['Label'].value_counts().to_string(), flush=True)
    absent = sorted(set(UNKNOWN) - set(df['Label']))
    if absent: raise ValueError(f'Unknown families absent after cleaning: {absent}')
    known = df.loc[~df['Label'].isin(UNKNOWN)].copy()
    unknown = df.loc[df['Label'].isin(UNKNOWN)].copy()
    if 'BENIGN' not in set(known['Label']): raise ValueError('No BENIGN rows')
    train, remainder = train_test_split(known, test_size=.30, stratify=known['Label'], random_state=SEED)
    val, test = train_test_split(remainder, test_size=.50, stratify=remainder['Label'], random_state=SEED)
    ae_train = train.loc[train['Label'].eq('BENIGN')]
    ae_val = val.loc[val['Label'].eq('BENIGN')]
    for name, frame in [('known_train', train), ('known_validation', val), ('known_test', test),
                        ('unknown_test', unknown), ('autoencoder_benign_train', ae_train),
                        ('autoencoder_benign_validation', ae_val)]:
        path = out / (name + '.parquet')
        frame.to_parquet(path, index=False)
        print('SAVED', path.name, frame.shape, flush=True)
    
    (out / 'feature_list.json').write_text(json.dumps(FEATURES, indent=2))
    counts = pd.concat({'train': train['Label'].value_counts(), 'validation': val['Label'].value_counts(),
                        'test': test['Label'].value_counts(), 'unknown_test': unknown['Label'].value_counts()}, axis=1).fillna(0).astype(int)
    counts.to_csv(out / 'split_summary.csv')
    config = {'dataset': 'Improved CICIDS2017', 'seed': SEED, 'unknown_families': list(UNKNOWN),
              'attempted_policy': 'Non-held-out attempted flows relabeled BENIGN; attempted flows of held-out families excluded',
              'v15_expected_features': len(FEATURES), 'known_rows': len(known), 'unknown_rows': len(unknown),
              'clean_rows': len(df), 'raw_rows': initial, 'excluded_unknown_family_attempted': excluded_unknown_attempted,
              'requires_v15_retraining': True, 'note': 'Do not reuse V15 model/scaler/threshold on new splits without proof of identical fit data.'}
    (out / 'experiment_config.json').write_text(json.dumps(config, indent=2))
    assert not any(x in set(train['Label']) or x in set(val['Label']) or x in set(test['Label']) for x in UNKNOWN)
    assert set(train.index).isdisjoint(set(val.index)) and set(train.index).isdisjoint(set(test.index)) and set(val.index).isdisjoint(set(test.index))
    assert np.isfinite(df[FEATURES].to_numpy(dtype=np.float64)).all()
    print('\nSUCCESS: GC2 shared Improved CICIDS2017 data prepared. V15 retraining required for this split.', flush=True)

if __name__ == '__main__':
    main()
