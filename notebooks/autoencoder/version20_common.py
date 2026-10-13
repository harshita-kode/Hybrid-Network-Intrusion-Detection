"""V16 shared helpers for all-column Improved CIC-IDS2017 exports.
Keep metadata in dataframes but NEVER include labels, IDs, timestamps, IPs or attempted flags in model input.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

VERSION = 'version20'
UNKNOWN = ('SSH-Patator', 'FTP-Patator', 'DoS Slowhttptest')
# Only these four fields are explicitly excluded from model features.
# All original columns remain in the loaded/saved dataframes.
EXCLUDE = {'label', 'attempted category', 'source_file', 'is_attempted', 'id'}

PORT_COLUMNS = ('Dst Port','Destination Port')

def validate_frame(df, name, benign_only=False):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    if df.columns.duplicated().any():
        raise ValueError(f'{name}: duplicate column names')
    if 'Label' not in df.columns:
        raise KeyError(f'{name}: Label is required')
    df['Label'] = df['Label'].astype(str).str.strip()
    if benign_only and not df['Label'].str.upper().eq('BENIGN').all():
        raise ValueError(f'{name}: non-BENIGN records detected')
    if 'is_attempted' in df.columns and df['is_attempted'].astype(str).str.lower().isin(['true','1']).any():
        raise ValueError(f'{name}: attempted records must not enter primary splits')
    return df

def numeric_feature_frame(df):
    """Keep every numeric original column except the four explicitly excluded fields.

    Non-numeric fields (e.g. IP addresses, timestamps, flow IDs) remain in the
    dataframe for analysis, but cannot be passed as raw strings to a Dense AE.
    """
    candidates = [c for c in df.columns if c.strip().casefold() not in EXCLUDE]
    usable, nonnumeric = {}, []
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]):
            usable[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            numeric = pd.to_numeric(df[c], errors='coerce')
            if numeric.notna().sum() and numeric.notna().mean() >= 0.99:
                usable[c] = numeric
            else:
                nonnumeric.append(c)
    if not usable:
        raise ValueError('No numeric flow features found')
    result = pd.DataFrame(usable, index=df.index).replace([np.inf, -np.inf], np.nan)
    result.attrs['non_numeric_columns'] = nonnumeric
    return result

def signed_log(frame):
    return np.sign(frame) * np.log1p(np.abs(frame))

def fit_preprocessing(train_df):
    x = numeric_feature_frame(train_df)
    columns = [c for c in x if x[c].notna().any() and x[c].nunique(dropna=True)>1]
    if not columns: raise ValueError('No nonconstant numerical training features')
    z = signed_log(x[columns])
    medians = z.median().fillna(0)
    scaler = RobustScaler().fit(z.fillna(medians))
    return {'columns':columns,'medians':medians,'scaler':scaler,'input_dim':len(columns),
            'excluded_fields':sorted(EXCLUDE),
            'non_numeric_columns':x.attrs.get('non_numeric_columns', []),
            'constant_or_empty_columns':[c for c in x if c not in columns]}

def transform(df, prep):
    missing = [c for c in prep['columns'] if c not in df.columns]
    if missing: raise KeyError(f'Missing model features: {missing}')
    x = df[prep['columns']].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan)
    z = signed_log(x).fillna(prep['medians'])
    result = np.clip(prep['scaler'].transform(z), -20, 20).astype('float32')
    if not np.isfinite(result).all(): raise ValueError('Nonfinite transformed inputs')
    return result

def ports(df):
    for name in PORT_COLUMNS:
        if name in df.columns:
            values = pd.to_numeric(df[name],errors='coerce')
            return values.to_numpy(dtype=float)
    return np.full(len(df),np.nan)

def service_group(port):
    if not np.isfinite(port) or not 0 <= port <= 65535: return 'unknown'
    port = int(port)
    if port in (20,21): return 'ftp'
    if port in (80,443,8080,8443): return 'web'
    if port == 22: return 'ssh'
    if port == 53: return 'dns'
    if port < 1024: return 'other_well_known'
    if port < 49152: return 'registered'
    return 'dynamic'

def reconstruction_mse(models, x, batch_size=4096):
    scores = np.zeros(len(x), dtype=np.float64)
    for m in models:
        y = m.predict(x,batch_size=batch_size,verbose=0)
        scores += np.mean((x-y)**2,axis=1)
    return scores/len(models)

def calibrate(scores, port_values, target_fpr=.03, min_port=500, min_group=1000):
    scores=np.asarray(scores)
    port_values=np.asarray(port_values,dtype=float)
    global_t=float(np.quantile(scores,1-target_fpr))
    port_ts={}; group_ts={}; port_counts={}; group_counts={}
    valid=np.isfinite(port_values)&(port_values>=0)&(port_values<=65535)
    for port in np.unique(port_values[valid].astype(int)):
        mask=valid & (port_values==port)
        n=int(mask.sum()); port_counts[str(port)]=n
        if n>=min_port: port_ts[str(port)]=float(np.quantile(scores[mask],1-target_fpr))
    groups=np.array([service_group(v) for v in port_values])
    for group in np.unique(groups):
        mask=groups==group; n=int(mask.sum()); group_counts[str(group)]=n
        if group!='unknown' and n>=min_group:
            group_ts[str(group)]=float(np.quantile(scores[mask],1-target_fpr))
    return {'global_threshold':global_t,'port_thresholds':port_ts,'group_thresholds':group_ts,
            'port_counts':port_counts,'group_counts':group_counts,'target_fpr':target_fpr,
            'min_port':min_port,'min_group':min_group}

def local_thresholds(port_values, calibration):
    result=[]
    for port in port_values:
        if not np.isfinite(port) or not 0<=port<=65535:
            result.append(calibration['global_threshold']); continue
        key=str(int(port))
        result.append(calibration['port_thresholds'].get(key,
            calibration['group_thresholds'].get(service_group(port),calibration['global_threshold'])))
    return np.asarray(result,dtype=float)


BEHAVIOURAL_FEATURES = (
    'src_connections_60s', 'dst_connections_60s',
    'src_dst_connections_60s', 'src_dst_port_connections_60s',
    'seconds_since_previous_src_connection',
    'src_ssh_connections_60s', 'src_dst_ssh_connections_60s',
    'src_ssh_connections_300s', 'src_dst_ssh_connections_300s',
    'seconds_since_previous_ssh_connection',
)

def add_behavioural_features(df, window_seconds=60):
    """Causal connection counts; preserves row order and original fields.

    SSH history updates ONLY on destination port 22. Missing prior SSH
    connection is encoded as 300 seconds. Current event excluded from counts.
    All history is local to the supplied dataframe (existing split limitation).
    """
    from collections import defaultdict, deque
    df = df.copy()
    required = ['Timestamp', 'Src IP', 'Dst IP']
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(f'Missing behavioural metadata: {missing}')
    port_col = 'Dst Port' if 'Dst Port' in df else 'Destination Port'
    if port_col not in df:
        raise ValueError('Missing destination port')
    ts = pd.to_datetime(df['Timestamp'], errors='coerce')
    if ts.isna().any():
        raise ValueError(f'Invalid timestamps: {int(ts.isna().sum())}')
    order = np.argsort(ts.to_numpy(), kind='stable')
    times = ts.astype('int64').to_numpy() / 1e9
    src = df['Src IP'].fillna('MISSING').astype(str).to_numpy()
    dst = df['Dst IP'].fillna('MISSING').astype(str).to_numpy()
    pnum = pd.to_numeric(df[port_col], errors='coerce').to_numpy()
    histories = [defaultdict(deque) for _ in range(4)]
    ssh_histories = [defaultdict(deque) for _ in range(4)]
    last_src = {}
    last_ssh_src = {}
    values = np.zeros((len(df), len(BEHAVIOURAL_FEATURES)), dtype=np.float64)
    for i in order:
        t = float(times[i])
        port_key = int(pnum[i]) if np.isfinite(pnum[i]) else -1
        keys = (src[i], dst[i], (src[i], dst[i]),
                (src[i], dst[i], port_key))
        for j, key in enumerate(keys):
            q = histories[j][key]
            while q and q[0] < t - window_seconds:
                q.popleft()
            values[i, j] = len(q)
        values[i, 4] = max(0., t - last_src[src[i]]) if src[i] in last_src else window_seconds
        for j, key in enumerate(keys):
            histories[j][key].append(t)
        last_src[src[i]] = t
        if port_key == 22:
            ssh_keys = (src[i], (src[i], dst[i]), src[i], (src[i], dst[i]))
            windows = (60, 60, 300, 300)
            for j, (key, window) in enumerate(zip(ssh_keys, windows)):
                q = ssh_histories[j][key]
                while q and q[0] < t - window:
                    q.popleft()
                values[i, 5+j] = len(q)
            values[i, 9] = max(0., t - last_ssh_src[src[i]]) if src[i] in last_ssh_src else 300.
            for j, key in enumerate(ssh_keys):
                ssh_histories[j][key].append(t)
            last_ssh_src[src[i]] = t
        else:
            # Non-SSH flows get neutral zeros, not counts of other services.
            values[i, 9] = 300.
    for j, name in enumerate(BEHAVIOURAL_FEATURES):
        df[name] = values[:, j]
    return df


def verify_behavioural_inputs(train_df, val_df, prep, x_train, x_val):
    """Check training/validation alignment and guard against metadata leakage."""
    expected = set(BEHAVIOURAL_FEATURES)
    selected = set(prep['columns'])
    missing = sorted(expected - selected)
    if missing:
        raise ValueError(f'Behavioural inputs missing from training: {missing}')
    forbidden = {'label', 'attempted category', 'source_file', 'is_attempted',
                 'id', 'flow id', 'src ip', 'dst ip', 'timestamp'}
    leaked = sorted(c for c in prep['columns'] if c.strip().casefold() in forbidden)
    if leaked:
        raise ValueError(f'Metadata/identifier leakage into model inputs: {leaked}')
    if x_train.shape[1] != x_val.shape[1] or x_train.shape[1] != prep['input_dim']:
        raise ValueError('Training/validation feature dimension mismatch')
    if len(train_df) != len(x_train) or len(val_df) != len(x_val):
        raise ValueError('Training/validation row count mismatch')
    if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
        raise ValueError('Non-finite transformed inputs')
    print('Training:', x_train.shape, '| Validation:', x_val.shape)
    print('Behavioural features included:', sorted(expected))
    print('Selected model columns:', prep['columns'])
    print('Non-numeric columns:', prep.get('non_numeric_columns', []))
    print('Constant/empty columns:', prep.get('constant_or_empty_columns', []))
    print('Feature verification passed.')


def load_clean_split(data_dir, name, benign_only=False):
    """Read shared CSV/Parquet without modifying it; remove attempted rows in memory."""
    from pathlib import Path
    folder = Path(data_dir)
    csv_path = folder / (name + '.csv')
    parquet_path = folder / (name + '.parquet')
    if csv_path.exists():
        frame = pd.read_csv(csv_path, low_memory=False)
        source = csv_path
    elif parquet_path.exists():
        frame = pd.read_parquet(parquet_path)
        source = parquet_path
    else:
        raise FileNotFoundError(f'Neither CSV nor Parquet exists for {name} in {folder}')
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.columns.duplicated().any():
        raise ValueError(f'{name}: duplicate column names')
    if 'Label' not in frame:
        raise KeyError(f'{name}: missing Label column')
    original_count = len(frame)
    if 'is_attempted' in frame:
        attempted = frame['is_attempted'].astype(str).str.strip().str.casefold().isin(
            ('true', '1', 'yes', 't'))
    else:
        attempted = pd.Series(False, index=frame.index)
        print(f'WARNING: {name} has no is_attempted field; attempted rows cannot be audited')
    # Do not silently reinterpret attack labels. Reject non-BENIGN rows in BENIGN splits.
    frame = frame.loc[~attempted].copy()
    print(f'{name}: read {original_count:,} from {source.name}; '
          f'removed {int(attempted.sum()):,} attempted; retained {len(frame):,}')
    frame = validate_frame(frame, name, benign_only=benign_only)
    if benign_only and not frame['Label'].str.upper().eq('BENIGN').all():
        raise ValueError(f'{name}: non-BENIGN records after filtering')
    if frame.empty:
        raise ValueError(f'{name}: empty after attempted-record filtering')
    return frame.reset_index(drop=True)
