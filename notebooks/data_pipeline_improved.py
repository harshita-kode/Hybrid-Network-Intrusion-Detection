from pathlib import Path
import json
import re
import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# GC2 - IMPROVED CICIDS2017 DATA PIPELINE
# VERSION: DATA_V3_FULL
#
# Purpose:
#   1. Preserve all original dataset columns.
#   2. Retain identifiers and attempted-attack metadata.
#   3. Clean the modelling features.
#   4. Remove conflicting-label patterns.
#   5. Remove duplicate feature vectors.
#   6. Hold out three unknown attack families.
#   7. Create shared XGBoost and Autoencoder datasets.
#
# IMPORTANT:
#   Preserving a column does not mean using it for ML.
# ============================================================


SEED = 42

UNKNOWN_FAMILIES = [
    "Botnet",
    "FTP-Patator",
    "DoS Slowhttptest",
]

EXPECTED_RESTORED_COLUMNS = [
    "Attempted Category",
    "Bwd Bulk Rate Avg",
    "Bwd Bytes/Bulk Avg",
    "Bwd PSH Flags",
    "Bwd Packet/Bulk Avg",
    "Bwd URG Flags",
    "Dst IP",
    "Flow ID",
    "Fwd Bulk Rate Avg",
    "Fwd Bytes/Bulk Avg",
    "Fwd Packet/Bulk Avg",
    "Src IP",
    "Src Port",
    "Timestamp",
    "id",
    "is_attempted",
]

IDENTIFIER_COLUMNS = {
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Src Port",
    "Timestamp",
    "id",
    "source_file",
}

LABEL_METADATA_COLUMNS = {
    "Label",
    "Attempted Category",
    "is_attempted",
    "Original_Label",
}

# These columns are retained in the exported datasets
# but excluded from the model input.
EXCLUDED_MODEL_COLUMNS = {
    "Dst Port",
    "Destination Port",
    "Fwd Header Length.1",
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Fwd Bytes/Bulk Avg",
    "Fwd Packet/Bulk Avg",
    "Fwd Bulk Rate Avg",
    "Bwd Bytes/Bulk Avg",
    "Bwd Packet/Bulk Avg",
    "Bwd Bulk Rate Avg",
}

NONNEGATIVE_COLUMNS = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Seg Size Min",
    "min_seg_size_forward",
]


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

while ROOT.name != "Hybrid-Network-Intrusion-Detection":
    if ROOT == ROOT.parent:
        raise RuntimeError("Run this script inside the GC2 repository.")
    ROOT = ROOT.parent

RAW_DIR = ROOT / "data" / "raw" / "improved_CIC-IDS2017"

OUT_DIR = ROOT / "data" / "processed" / "Data_V3_full"

OUT_DIR.mkdir(parents=True, exist_ok=True)

if not RAW_DIR.exists():
    raise FileNotFoundError(f"Improved dataset directory not found: {RAW_DIR}")

print("=" * 70)
print("GC2 IMPROVED CICIDS2017 - FULL COLUMN PIPELINE")
print("=" * 70)

print("Input:", RAW_DIR)
print("Output:", OUT_DIR)


# ============================================================
# 2. LOAD ALL RAW CSV FILES
# ============================================================

print("\n[1/12] Loading raw improved dataset...")

csv_files = sorted(RAW_DIR.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found in improved dataset directory.")

print("CSV files found:", len(csv_files))

frames = []
reference_columns = None

for csv_file in csv_files:

    print("Loading:", csv_file.name)

    frame = pd.read_csv(csv_file, low_memory=False)

    frame.columns = frame.columns.astype(str).str.strip()

    if frame.columns.duplicated().any():
        raise ValueError(f"Duplicate column names in {csv_file.name}")

    current_columns = set(frame.columns)

    if reference_columns is None:
        reference_columns = current_columns

    elif current_columns != reference_columns:
        raise ValueError(f"Schema mismatch in {csv_file.name}")

    frame["source_file"] = csv_file.stem

    print(f"Loaded {csv_file.name}: " f"{len(frame):,} rows")

    frames.append(frame)

data = pd.concat(frames, ignore_index=True)

del frames
gc.collect()

print("\nMerged shape:", data.shape)

if "Label" not in data.columns:
    raise ValueError("Label column is missing.")

raw_column_names = list(data.columns)


# ============================================================
# 3. STANDARDIZE LABELS
# ============================================================

print("\n[2/12] Standardizing labels...")

data["Label"] = data["Label"].astype(str).str.strip()

# Preserve original labels for auditing.
data["Original_Label"] = data["Label"]


def normalize_label(label):

    label = str(label).strip()

    label = re.sub(
        r"^Web Attack\s*[-–—�]\s*", "Web Attack - ", label, flags=re.IGNORECASE
    )

    label = re.sub(r"\s+", " ", label).strip()

    attempted = bool(re.search(r"\s*-\s*Attempted$", label, flags=re.IGNORECASE))

    base = re.sub(r"\s*-\s*Attempted$", "", label, flags=re.IGNORECASE).strip()

    aliases = {
        "bot": "Botnet",
        "botnet": "Botnet",
        "ftp-patator": "FTP-Patator",
        "ftp patator": "FTP-Patator",
        "ftp-bruteforce": "FTP-Patator",
        "ftp-brute force": "FTP-Patator",
        "dos slowhttptest": "DoS Slowhttptest",
        "dos slowhttp test": "DoS Slowhttptest",
        "dos slow http test": "DoS Slowhttptest",
    }
    s = alias.get(s.casefold(), s)
    return s, attempted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/raw/improved_CIC-IDS2017",
        help="Folder containing extracted Improved CICIDS2017 CSV files",
    )
    ap.add_argument("--out", default="data/processed/Data_V3")
    args = ap.parse_args()
    source, out = Path(args.input).resolve(), Path(args.out).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Extracted dataset folder not found: {source}")
    names = sorted(
        p
        for p in source.rglob("*.csv")
        if not p.name.startswith("._") and "__MACOSX" not in p.parts
    )
    if not names:
        raise ValueError(f"No CSV files found inside {source}")
    out.mkdir(parents=True, exist_ok=True)
    print("INPUT FOLDER:", source, "\nOUTPUT:", out, flush=True)
    print("CSV FILES:", len(names), flush=True)
    frames = []
    for path in names:
        print("Loading:", path.relative_to(source), flush=True)
        part = pd.read_csv(path, low_memory=False)
        part.columns = part.columns.astype(str).str.strip()
        if part.columns.duplicated().any():
            raise ValueError(f"Duplicate headers: {path}")
        missing = set(FEATURES + ["Label"]) - set(part.columns)
        if missing:
            raise ValueError(
                f"{path.name}: missing required V15 columns: {sorted(missing)}"
            )
        part["source_file"] = path.stem
        frames.append(part)
        print(f"Loaded {path.name}: {len(part):,} rows", flush=True)
    df = pd.concat(frames, ignore_index=True)
    del frames
    initial = len(df)
    print("MERGED:", df.shape, flush=True)
    canonical = df["Label"].map(canonical_label)
    df["Label"] = canonical.map(lambda p: p[0])
    attempted = canonical.map(lambda p: p[1]).to_numpy(dtype=bool)
    if "Attempted Category" in df.columns:
        ac = pd.to_numeric(df["Attempted Category"], errors="coerce")
        attempted |= (ac.notna() & ac.ne(-1)).to_numpy()
    df["is_attempted"] = attempted
    print("Original family counts (before attempted policy):")
    print(df["Label"].value_counts().to_string(), flush=True)
    print("Attempted rows:", int(attempted.sum()), flush=True)
    # Audit attempted family before changing labels.
    audit = pd.crosstab(df["Label"], df["is_attempted"])
    audit.to_csv(out / "attempted_audit.csv")
    unknown_family_attempted = df["Label"].isin(UNKNOWN) & df["is_attempted"]
    excluded_unknown_attempted = int(unknown_family_attempted.sum())
    df = df.loc[~unknown_family_attempted].copy()
    df.loc[df["is_attempted"], "Label"] = "BENIGN"
    required = FEATURES + ["Label", "source_file"]
    if "Dst Port" in df.columns:
        required.append("Dst Port")
    df = df[required].copy()
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    vals = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    bad = vals.isna().any(axis=1)
    removed_nonfinite = int(bad.sum())
    df = df.loc[~bad].copy()
    negative_cols = [
        c
        for c in [
            "Flow Duration",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Fwd Header Length",
            "Bwd Header Length",
            "Fwd Seg Size Min",
        ]
        if c in df
    ]
    invalid_negative = (df[negative_cols] < 0).any(axis=1)
    removed_negative = int(invalid_negative.sum())
    df = df.loc[~invalid_negative].copy()

    identity = FEATURES + (["Dst Port"] if "Dst Port" in df else [])
    h = pd.util.hash_pandas_object(df[identity], index=False)
    label_counts = df.groupby(h, sort=False)["Label"].nunique()
    bad_hashes = label_counts.index[label_counts > 1]
    conflict_rows = int(h.isin(bad_hashes).sum())
    df = df.loc[~h.isin(bad_hashes)].copy()
    before_dup = len(df)
    df = df.drop_duplicates(subset=identity).reset_index(drop=True)
    duplicates = before_dup - len(df)
    print(
        "Removed nonfinite:",
        removed_nonfinite,
        "invalid negatives:",
        removed_negative,
        "conflicting rows:",
        conflict_rows,
        "duplicates:",
        duplicates,
        flush=True,
    )
    print(
        f"Saved {name}.parquet: "
        f"{len(frame):,} rows x "
        f"{len(frame.columns)} columns"
    )

with open(OUT_DIR / "feature_list.json", "w") as f:

    json.dump(model_features, f, indent=2)

config = {
    "dataset": "Improved CICIDS2017",
    "version": "Data_V3_full",
    "seed": SEED,
    "unknown_classes": UNKNOWN_FAMILIES,
    "known_classes": sorted(known_df["Label"].unique().tolist()),
    "attempted_policy": (
        "Known-family attempted flows relabelled BENIGN; "
        "unknown-family attempted flows excluded"
    ),
    "full_column_count": len(data.columns),
    "model_feature_count": len(model_features),
    "split_ratio": [0.70, 0.15, 0.15],
}

with open(OUT_DIR / "experiment_config.json", "w") as f:

    json.dump(config, f, indent=2)

split_summary = (
    pd.concat(
        {
            "Train": known_train["Label"].value_counts(),
            "Validation": known_validation["Label"].value_counts(),
            "Test": known_test["Label"].value_counts(),
            "Unknown_test": unknown_df["Label"].value_counts(),
        },
        axis=1,
    )
    .fillna(0)
    .astype(int)
)

split_summary.to_csv(OUT_DIR / "split_summary.csv")

# Save full schema for auditing.
pd.DataFrame(
    {
        "Column": data.columns,
        "Used_for_ML": [col in model_features for col in data.columns],
        "Data_type": [str(data[col].dtype) for col in data.columns],
    }
).to_csv(OUT_DIR / "column_schema.csv", index=False)

print("\n" + "=" * 70)
print("GC2 DATA_V3_FULL PIPELINE COMPLETED")
print("=" * 70)

print("Final records:", len(data))
print("Full columns:", len(data.columns))
print("Model features:", len(model_features))
print("Known classes:", known_df["Label"].nunique())
print("Unknown classes:", unknown_df["Label"].nunique())
print("Output folder:", OUT_DIR)

print("\nIMPORTANT:")
print(
    "All original columns are retained for auditing. "
    "Models must use feature_list.json, not every column."
)
