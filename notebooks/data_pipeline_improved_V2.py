from pathlib import Path
import json
import re
import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# GC2 - IMPROVED CIC-IDS2017 FINAL DATA PIPELINE
# ============================================================

SEED = 42

UNKNOWN_FAMILIES = [
    "SSH-Patator",
    "FTP-Patator",
    "DoS Slowhttptest",
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
    "Original_Label",
    "Attempted Category",
    "is_attempted",
}

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

REQUIRED_COLUMNS = [
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


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

while ROOT.name != "Hybrid-Network-Intrusion-Detection":
    if ROOT == ROOT.parent:
        raise RuntimeError("Script must be inside GC2 repository")
    ROOT = ROOT.parent

RAW_DIR = ROOT / "data/raw/improved_CIC-IDS2017"

OUT_DIR = ROOT / "data/processed/Data_V3_SSH_Unknown_final"

if not RAW_DIR.exists():
    raise FileNotFoundError(RAW_DIR)

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("INPUT:", RAW_DIR)
print("OUTPUT:", OUT_DIR)


# ============================================================
# 2. LOAD DATASET
# ============================================================

csv_files = sorted(RAW_DIR.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found")

frames = []
reference_columns = None

for file in csv_files:

    df = pd.read_csv(file, low_memory=False)

    df.columns = df.columns.astype(str).str.strip()

    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate columns: {file.name}")

    if reference_columns is None:
        reference_columns = set(df.columns)

    elif set(df.columns) != reference_columns:
        raise ValueError(f"Schema mismatch: {file.name}")

    df["source_file"] = file.stem

    frames.append(df)

    print(f"Loaded {file.name}: {len(df):,}")

data = pd.concat(frames, ignore_index=True)

del frames
gc.collect()

original_rows = len(data)
original_columns = list(data.columns)

print("MERGED:", data.shape)

if "Label" not in data.columns:
    raise ValueError("Label column missing")


# ============================================================
# 3. STANDARDIZE LABELS
# ============================================================

data["Original_Label"] = data["Label"].astype(str).str.strip()


def normalize_label(value):

    label = re.sub(r"\s+", " ", str(value).strip())

    label = re.sub(
        r"^Web Attack\s*[-–—�]\s*", "Web Attack - ", label, flags=re.IGNORECASE
    )

    attempted = bool(re.search(r"\s*-\s*Attempted$", label, flags=re.IGNORECASE))

    base = re.sub(r"\s*-\s*Attempted$", "", label, flags=re.IGNORECASE).strip()

    aliases = {
        "bot": "Botnet",
        "botnet": "Botnet",
        "ftp patator": "FTP-Patator",
        "ftp-patator": "FTP-Patator",
        "ftp-bruteforce": "FTP-Patator",
        "dos slowhttptest": "DoS Slowhttptest",
        "dos slowhttp test": "DoS Slowhttptest",
        "dos slow http test": "DoS Slowhttptest",
    }

    base = aliases.get(base.casefold(), base)

    if attempted:
        return base + " - Attempted"

    return base


data["Label"] = data["Original_Label"].map(normalize_label)

print("\nORIGINAL LABEL DISTRIBUTION")
print(data["Label"].value_counts())


# ============================================================
# 4. ATTEMPTED ATTACK HANDLING
# ============================================================

attempted_mask = data["Label"].str.contains(
    r"\s*-\s*Attempted$", case=False, regex=True, na=False
)

if "Attempted Category" in data.columns:

    attempted_category = pd.to_numeric(data["Attempted Category"], errors="coerce")

    attempted_mask |= attempted_category.notna() & attempted_category.ne(-1)

data["is_attempted"] = attempted_mask.astype(bool)

base_labels = data["Label"].str.replace(
    r"\s*-\s*Attempted$", "", regex=True, flags=re.IGNORECASE
)

attempted_unknown_mask = attempted_mask & base_labels.isin(UNKNOWN_FAMILIES)

attempted_audit = (
    data.loc[attempted_mask, "Original_Label"]
    .value_counts()
    .rename_axis("Original_Label")
    .reset_index(name="count")
)

attempted_audit.to_csv(OUT_DIR / "attempted_audit.csv", index=False)

print("Attempted rows:", int(attempted_mask.sum()))

print("Held-out attempted rows:", int(attempted_unknown_mask.sum()))

# Known-family attempted flows become BENIGN.
data.loc[attempted_mask & ~attempted_unknown_mask, "Label"] = "BENIGN"

# Preserve held-out family identity until exclusion.
data.loc[attempted_unknown_mask, "Label"] = base_labels.loc[attempted_unknown_mask]


# ============================================================
# 5. NUMERIC FEATURES
# ============================================================

# Full traffic signature for duplicate detection.
dedup_features = [
    col
    for col in data.columns
    if col not in (IDENTIFIER_COLUMNS | LABEL_METADATA_COLUMNS)
]

for col in dedup_features:

    data[col] = pd.to_numeric(data[col], errors="coerce")

data[dedup_features] = data[dedup_features].replace([np.inf, -np.inf], np.nan)

# Actual model features exclude redundant columns.
candidate_features = [
    col for col in dedup_features if col not in EXCLUDED_MODEL_COLUMNS
]

print("Deduplication features:", len(dedup_features))
print("Candidate ML features:", len(candidate_features))


# ============================================================
# 6. REMOVE INVALID RECORDS
# ============================================================

before = len(data)

data = data.dropna(subset=dedup_features).copy()

removed_nonfinite = before - len(data)

before = len(data)

for col in NONNEGATIVE_COLUMNS:

    if col in data.columns:

        data = data.loc[data[col] >= 0].copy()

removed_negative = before - len(data)

print("Removed nonfinite:", removed_nonfinite)
print("Removed invalid negatives:", removed_negative)


# ============================================================
# 7. EXCLUDE ATTEMPTED UNKNOWN FAMILIES
# ============================================================

# Must happen BEFORE conflict detection.

before = len(data)

data = data.loc[~(data["is_attempted"] & data["Label"].isin(UNKNOWN_FAMILIES))].copy()

removed_unknown_attempted = before - len(data)

print("Excluded attempted unknown-family rows:", removed_unknown_attempted)


# ============================================================
# 8. REMOVE CONFLICTING LABEL PATTERNS
# ============================================================

hashes = pd.util.hash_pandas_object(data[dedup_features], index=False)

conflicts = data.groupby(hashes, sort=False)["Label"].nunique()

conflicting_hashes = conflicts[conflicts > 1].index

before = len(data)

data = data.loc[~hashes.isin(conflicting_hashes)].copy()

removed_conflicts = before - len(data)

print("Removed conflicting rows:", removed_conflicts)


# ============================================================
# 9. REMOVE DUPLICATES
# ============================================================

before = len(data)

data = data.drop_duplicates(subset=dedup_features).copy()

removed_duplicates = before - len(data)

print("Removed duplicates:", removed_duplicates)


# ============================================================
# 10. VERIFY ORIGINAL COLUMNS
# ============================================================

missing_original = [col for col in original_columns if col not in data.columns]

missing_required = [col for col in REQUIRED_COLUMNS if col not in data.columns]

if missing_original or missing_required:
    raise ValueError(f"Missing columns: " f"{missing_original + missing_required}")

data = data.reset_index(drop=True)

print("\nCLEAN MASTER:", data.shape)
print(data["Label"].value_counts())


# ============================================================
# 11. KNOWN / UNKNOWN SPLIT
# ============================================================

unknown_df = data.loc[data["Label"].isin(UNKNOWN_FAMILIES)].copy()

known_df = data.loc[~data["Label"].isin(UNKNOWN_FAMILIES)].copy()

for attack in UNKNOWN_FAMILIES:

    if not unknown_df["Label"].eq(attack).any():
        raise ValueError(f"Unknown class missing: {attack}")

known_train, temp = train_test_split(
    known_df, test_size=0.30, random_state=SEED, stratify=known_df["Label"]
)

known_validation, known_test = train_test_split(
    temp, test_size=0.50, random_state=SEED, stratify=temp["Label"]
)


# ============================================================
# 12. AUTOENCODER DATASETS
# ============================================================

# Only originally BENIGN records.
# Relabelled attempted attacks are excluded from AE fitting.

ae_train = known_train.loc[
    known_train["Label"].eq("BENIGN") & ~known_train["is_attempted"]
].copy()

ae_validation = known_validation.loc[
    known_validation["Label"].eq("BENIGN") & ~known_validation["is_attempted"]
].copy()


# ============================================================
# 13. FINAL MODEL FEATURES
# ============================================================

# Determine constant features using training data only.

model_features = [
    col for col in candidate_features if known_train[col].nunique(dropna=False) > 1
]

assert not any(
    col in (IDENTIFIER_COLUMNS | LABEL_METADATA_COLUMNS | EXCLUDED_MODEL_COLUMNS)
    for col in model_features
)

print("Final model features:", len(model_features))


# ============================================================
# 14. FINAL VALIDATION
# ============================================================

assert len(data) == len(known_df) + len(unknown_df)

assert not known_train["Label"].isin(UNKNOWN_FAMILIES).any()

assert not known_validation["Label"].isin(UNKNOWN_FAMILIES).any()

assert not known_test["Label"].isin(UNKNOWN_FAMILIES).any()

assert ae_train["Label"].eq("BENIGN").all()
assert ae_validation["Label"].eq("BENIGN").all()

assert not ae_train["is_attempted"].any()
assert not ae_validation["is_attempted"].any()

assert set(known_train.index).isdisjoint(known_validation.index)

assert set(known_train.index).isdisjoint(known_test.index)

assert set(known_validation.index).isdisjoint(known_test.index)

assert data.duplicated(subset=dedup_features).sum() == 0

assert np.isfinite(data[dedup_features].to_numpy(dtype=np.float64)).all()

print("\nALL VALIDATION CHECKS PASSED")


# ============================================================
# 15. SAVE DATASETS
# ============================================================

outputs = {
    "clean_master": data,
    "known_train": known_train,
    "known_validation": known_validation,
    "known_test": known_test,
    "unknown_test": unknown_df,
    "autoencoder_benign_train": ae_train,
    "autoencoder_benign_validation": ae_validation,
}

for name, frame in outputs.items():

    path = OUT_DIR / f"{name}.parquet"

    frame.to_parquet(path, index=False)

    print(f"SAVED {name}: " f"{len(frame):,} rows × " f"{len(frame.columns)} columns")


# ============================================================
# 16. SAVE FEATURE LIST AND CONFIGURATION
# ============================================================

with open(OUT_DIR / "feature_list.json", "w") as f:

    json.dump(model_features, f, indent=2)

config = {
    "dataset": "Improved CICIDS2017",
    "version": "Data_V3_SSH_Unknown_final",
    "seed": SEED,
    "unknown_classes": UNKNOWN_FAMILIES,
    "known_classes": sorted(known_df["Label"].unique().tolist()),
    "split": [0.70, 0.15, 0.15],
    "attempted_policy": (
        "Known-family attempted relabelled BENIGN; "
        "unknown-family attempted excluded; "
        "AE training uses originally BENIGN only"
    ),
    "full_column_count": len(data.columns),
    "model_feature_count": len(model_features),
    "raw_rows": original_rows,
    "clean_rows": len(data),
    "removed_nonfinite": removed_nonfinite,
    "removed_invalid_negative": removed_negative,
    "excluded_unknown_attempted": removed_unknown_attempted,
    "removed_conflicts": removed_conflicts,
    "removed_duplicates": removed_duplicates,
}

with open(OUT_DIR / "experiment_config.json", "w") as f:

    json.dump(config, f, indent=2)


# ============================================================
# 17. SPLIT SUMMARY
# ============================================================

summary = (
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

summary.to_csv(OUT_DIR / "split_summary.csv", index_label="Label")


# ============================================================
# 18. COLUMN SCHEMA
# ============================================================

pd.DataFrame(
    {
        "Column": data.columns,
        "Used_for_ML": [col in model_features for col in data.columns],
        "Data_type": [str(data[col].dtype) for col in data.columns],
    }
).to_csv(OUT_DIR / "column_schema.csv", index=False)


# ============================================================
# 19. FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("GC2 IMPROVED DATA PREPARATION COMPLETED")
print("=" * 65)

print("Original records:", original_rows)
print("Clean records:", len(data))
print("Full columns:", len(data.columns))
print("Model features:", len(model_features))

print("\nKNOWN CLASSES:")
print(known_df["Label"].value_counts())

print("\nUNKNOWN CLASSES:")
print(unknown_df["Label"].value_counts())

print("\nAUTOENCODER:")
print("Train:", ae_train.shape)
print("Validation:", ae_validation.shape)
print("Known test:", known_test.shape)
print("Unknown test:", unknown_df.shape)

print("\nOUTPUT:", OUT_DIR)

print("\nSPLIT SUMMARY:")
print(summary)

print("\nIMPORTANT:")
print(
    "Review the final class distribution before "
    "releasing these datasets for model training."
)
# ============================================================
# 20. COMPLETE DATASET DOCUMENTATION AND LOGGING
# ============================================================

from datetime import datetime
from contextlib import redirect_stdout

LOG_PATH = OUT_DIR / "GC2_Data_V3_Complete_Log.txt"


def generate_complete_log():

    print("=" * 80)
    print("GC2 - IMPROVED CIC-IDS2017")
    print("COMPLETE DATA PREPROCESSING AND DATASET DOCUMENTATION")
    print("=" * 80)

    print("\nGenerated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print("\nDATASET VERSION:")
    print("Data_V3_SSH_Unknown_final")

    print("\nSOURCE DATASET:")
    print("Improved CIC-IDS2017")

    print("\nSOURCE DIRECTORY:")
    print(RAW_DIR)

    print("\nOUTPUT DIRECTORY:")
    print(OUT_DIR)

    # --------------------------------------------------------
    # 1. SOURCE FILES
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("1. ORIGINAL SOURCE FILES")
    print("=" * 80)

    for file in csv_files:
        print(file.name)

    print("\nOriginal integrated records:", original_rows)
    print("Original columns:", len(original_columns))

    # --------------------------------------------------------
    # 2. PREPROCESSING SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("2. PREPROCESSING SUMMARY")
    print("=" * 80)

    print("Original records:", original_rows)
    print("Removed NaN/Inf:", removed_nonfinite)
    print("Removed invalid negatives:", removed_negative)
    print("Excluded attempted unknown-family:", removed_unknown_attempted)
    print("Removed conflicting labels:", removed_conflicts)
    print("Removed duplicate feature vectors:", removed_duplicates)

    print("\nFinal clean records:", len(data))
    print("Final columns:", len(data.columns))
    print("Model features:", len(model_features))

    print("\nTotal removed:", original_rows - len(data))

    print("Retention percentage:", round(len(data) / original_rows * 100, 4), "%")

    # --------------------------------------------------------
    # 3. UNKNOWN ATTACK CONFIGURATION
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("3. KNOWN AND UNKNOWN ATTACK CONFIGURATION")
    print("=" * 80)

    print("\nUNKNOWN ATTACK CLASSES:")

    for label in UNKNOWN_FAMILIES:
        print("-", label)

    print("\nKNOWN ATTACK CLASSES:")

    for label in sorted(known_df["Label"].unique()):
        print("-", label)

    print("\nUnknown attack records:", len(unknown_df))
    print("Known records:", len(known_df))

    # --------------------------------------------------------
    # 4. ALL DATASET FILES
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("4. GENERATED DATASETS")
    print("=" * 80)

    for name, frame in outputs.items():

        print("\n" + "-" * 70)
        print("FILE:", name + ".parquet")
        print("-" * 70)

        print("Purpose:")

        descriptions = {
            "clean_master": "Complete cleaned Improved CICIDS2017 dataset.",
            "known_train": "XGBoost supervised training dataset.",
            "known_validation": "XGBoost validation and hyperparameter tuning.",
            "known_test": "Final known-class evaluation for XGBoost and Autoencoder.",
            "unknown_test": "Held-out unknown attack evaluation.",
            "autoencoder_benign_train": "BENIGN-only Autoencoder training.",
            "autoencoder_benign_validation": "BENIGN-only Autoencoder threshold calibration.",
        }

        print(descriptions.get(name, "Dataset"))

        print("\nRows:", len(frame))
        print("Columns:", len(frame.columns))

        print("\nLABEL DISTRIBUTION:")

        print(frame["Label"].value_counts().to_string())

        print("\nALL COLUMNS:")

        for i, column in enumerate(frame.columns, start=1):

            print(f"{i:03d}. {column} " f"[{frame[column].dtype}]")

        print("\nMODEL FEATURES PRESENT:")

        available_features = [
            feature for feature in model_features if feature in frame.columns
        ]

        for i, feature in enumerate(available_features, start=1):
            print(f"{i:03d}. {feature}")

        print("\nMissing model features:")

        missing = [
            feature for feature in model_features if feature not in frame.columns
        ]

        print(missing if missing else "None")

        print("\nAttempted records:")

        if "is_attempted" in frame.columns:
            print(int(frame["is_attempted"].sum()))
        else:
            print("Column unavailable")

    # --------------------------------------------------------
    # 5. MODEL FEATURE CONTRACT
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("5. OFFICIAL MODEL FEATURE LIST")
    print("=" * 80)

    print("\nTotal model features:", len(model_features))

    for i, feature in enumerate(model_features, start=1):
        print(f"{i:03d}. {feature}")

    print("\nExcluded metadata columns:")

    for col in sorted(
        IDENTIFIER_COLUMNS | LABEL_METADATA_COLUMNS | EXCLUDED_MODEL_COLUMNS
    ):
        print("-", col)

    # --------------------------------------------------------
    # 6. SPLIT SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("6. TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 80)

    print("\nKnown training:", len(known_train))
    print("Known validation:", len(known_validation))
    print("Known testing:", len(known_test))

    print("\nAutoencoder BENIGN training:", len(ae_train))
    print("Autoencoder BENIGN validation:", len(ae_validation))

    print("\nUnknown attack testing:", len(unknown_df))

    print("\nCLASS-WISE SPLIT SUMMARY:")

    print(summary.to_string())

    # --------------------------------------------------------
    # 7. DATA QUALITY
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("7. FINAL DATA QUALITY VALIDATION")
    print("=" * 80)

    print("Missing model values:", int(data[model_features].isna().sum().sum()))

    print(
        "Infinite model values:",
        int(np.isinf(data[model_features].to_numpy(dtype=np.float64)).sum()),
    )

    print(
        "Duplicate feature vectors:", int(data.duplicated(subset=dedup_features).sum())
    )

    print("\nUnknown classes in training:")

    print(int(known_train["Label"].isin(UNKNOWN_FAMILIES).sum()))

    print("\nUnknown classes in validation:")

    print(int(known_validation["Label"].isin(UNKNOWN_FAMILIES).sum()))

    # --------------------------------------------------------
    # 8. FINAL STATUS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("8. FINAL STATUS")
    print("=" * 80)

    print("Preprocessing completed.")
    print("All generated datasets documented.")
    print("Original columns retained.")
    print("Model features separately identified.")
    print("Known/unknown separation validated.")

    print("\nIMPORTANT:")
    print(
        "Label, attempted metadata, IP addresses, timestamps "
        "and identifiers must not be used as model inputs."
    )

    print("\n" + "=" * 80)
    print("END OF GC2 DATA DOCUMENTATION")
    print("=" * 80)


# Write all documentation into one TXT file
with open(LOG_PATH, "w", encoding="utf-8") as log_file:

    with redirect_stdout(log_file):
        generate_complete_log()

print("\nComplete documentation saved:")
print(LOG_PATH)
