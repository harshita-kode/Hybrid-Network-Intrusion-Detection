"""Shared V15 feature preparation and reconstruction utilities. No repeated data cleaning."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

VERSION = "v15"
LABELS = ("BENIGN", "Botnet", "FTP-Patator", "DoS Slowhttptest")
FEATURES = None  # Discover numeric features from the training CSV, not the original CICIDS2017 schema.
# These columns are metadata, labels, or identifiers, not model inputs.
META = {"Label", "source_file", "Attempted Category", "id", "Flow ID",
        "Src IP", "Dst IP", "Src Port", "Timestamp", "Dst Port", "Destination Port", "Original_Label"}
PORT_COLUMNS = ("Dst Port", "Destination Port")

# Exact numeric schema supplied for normal_train.csv. Preserve its order.
TRAINING_FEATURES = """Protocol,Flow Duration,Total Fwd Packet,Total Bwd packets,Total Length of Fwd Packet,Total Length of Bwd Packet,Fwd Packet Length Max,Fwd Packet Length Min,Fwd Packet Length Mean,Fwd Packet Length Std,Bwd Packet Length Max,Bwd Packet Length Min,Bwd Packet Length Mean,Bwd Packet Length Std,Flow Bytes/s,Flow Packets/s,Flow IAT Mean,Flow IAT Std,Flow IAT Max,Flow IAT Min,Fwd IAT Total,Fwd IAT Mean,Fwd IAT Std,Fwd IAT Max,Fwd IAT Min,Bwd IAT Total,Bwd IAT Mean,Bwd IAT Std,Bwd IAT Max,Bwd IAT Min,Fwd PSH Flags,Fwd URG Flags,Fwd RST Flags,Bwd RST Flags,Fwd Header Length,Bwd Header Length,Fwd Packets/s,Bwd Packets/s,Packet Length Min,Packet Length Max,Packet Length Mean,Packet Length Std,Packet Length Variance,FIN Flag Count,SYN Flag Count,RST Flag Count,PSH Flag Count,ACK Flag Count,URG Flag Count,CWR Flag Count,ECE Flag Count,Down/Up Ratio,Average Packet Size,Fwd Segment Size Avg,Bwd Segment Size Avg,Subflow Fwd Packets,Subflow Fwd Bytes,Subflow Bwd Packets,Subflow Bwd Bytes,FWD Init Win Bytes,Bwd Init Win Bytes,Fwd Act Data Pkts,Fwd Seg Size Min,Active Mean,Active Std,Active Max,Active Min,Idle Mean,Idle Std,Idle Max,Idle Min,ICMP Code,ICMP Type,Total TCP Flow Time""".split(",")


def validate_frame(df, name, benign_only=False):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    if df.columns.duplicated().any():
        raise ValueError(f"{name}: duplicate column names")
    if "Label" not in df.columns:
        raise ValueError(f"{name}: missing Label column")
    if benign_only and not df["Label"].eq("BENIGN").all():
        raise ValueError(f"{name}: contains non-BENIGN rows")
    missing = [col for col in TRAINING_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing expected processed features: {missing}")
    numeric = df[TRAINING_FEATURES].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return df, numeric


def _port(df):
    for col in PORT_COLUMNS:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 65535).to_numpy(dtype=int)
    return None


def service_group(port):
    port = int(port)
    if port in (20, 21): return "ftp"
    if port in (80, 443, 8080, 8443): return "web"
    if port == 22: return "ssh"
    if port == 53: return "dns"
    if port < 1024: return "other_well_known"
    if port < 49152: return "registered"
    return "dynamic"


def _engineer(x, feature_columns):
    numeric = x.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return np.sign(numeric) * np.log1p(np.abs(numeric))


def fit_preprocessing(x):
    # No repeated dataset-level cleaning or feature selection from Notebook 01.
    # Drop only training-constant / entirely empty columns, fitted on train alone.
    columns = [c for c in TRAINING_FEATURES if x[c].notna().any()
               and x[c].nunique(dropna=True) > 1]
    if not columns:
        raise ValueError("No usable model features")
    engineered = _engineer(x, columns)
    medians = engineered.median().fillna(0)
    scaler = RobustScaler().fit(engineered.fillna(medians).fillna(0))
    return {"columns": columns, "engineered_columns": columns,
            "medians": medians, "scaler": scaler}


def transform(x, prep):
    missing = [c for c in prep["columns"] if c not in x.columns]
    if missing:
        raise ValueError(f"Validation/testing missing training features: {missing}")
    engineered = _engineer(x, prep["columns"])
    values = prep["scaler"].transform(engineered.fillna(prep["medians"]).fillna(0))
    values = np.clip(values, -20, 20).astype("float32")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite model inputs")
    return values


def reconstruction_mse(models,x,batch_size=4096):
    total=np.zeros(len(x),dtype=np.float64)
    for model in models:
        pred=model.predict(x,batch_size=batch_size,verbose=0)
        total+=np.mean((x-pred)**2,axis=1)
    return total/len(models)

def benign_threshold(scores,target_fpr):
    return float(np.quantile(scores,1-target_fpr))

def thresholds_for_ports(ports,calibration):
    results=[]
    for p in ports:
        if pd.isna(p) or not (0 <= float(p) <= 65535):
            results.append(calibration["global_threshold"]); continue
        p=int(p); group=service_group(p)
        results.append(calibration["port_thresholds"].get(str(p),calibration["group_thresholds"].get(group,calibration["global_threshold"])))
    return np.asarray(results,dtype=float)
