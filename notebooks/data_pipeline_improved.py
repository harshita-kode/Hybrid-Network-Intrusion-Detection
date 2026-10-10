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

    base = aliases.get(base.casefold(), base)

    if attempted:
        return base + " - Attempted"

    return base


data["Label"] = data["Label"].map(normalize_label)

print("\nOriginal standardized class distribution:")
print(data["Label"].value_counts())


# ============================================================
# 4. ATTEMPTED ATTACK HANDLING
# ============================================================

print("\n[3/12] Auditing attempted attacks...")

attempted_mask = data["Label"].str.contains(r"\s*-\s*Attempted$", case=False, na=False)

if "Attempted Category" in data.columns:

    attempted_category = pd.to_numeric(data["Attempted Category"], errors="coerce")

    attempted_mask |= attempted_category.notna() & attempted_category.ne(-1)

# Preserve attempted status as metadata.
data["is_attempted"] = attempted_mask.astype(bool)

print("Total attempted records:", int(attempted_mask.sum()))

attempted_audit = data.loc[attempted_mask, "Original_Label"].value_counts()

attempted_audit.to_csv(OUT_DIR / "attempted_audit.csv")

# Identify attempted flows belonging to held-out families.
base_labels = data["Label"].str.replace(
    r"\s*-\s*Attempted$", "", regex=True, flags=re.IGNORECASE
)

attempted_unknown_mask = attempted_mask & base_labels.isin(UNKNOWN_FAMILIES)

print("Attempted held-out-family records:", int(attempted_unknown_mask.sum()))

# Match the previous V3 experiment:
# attempted flows from known families are treated as BENIGN.
# attempted flows from unknown families are excluded later.
data.loc[attempted_mask & ~attempted_unknown_mask, "Label"] = "BENIGN"

# Keep unknown attempted records identifiable until exclusion.
data.loc[attempted_unknown_mask, "Label"] = base_labels.loc[attempted_unknown_mask]


# ============================================================
# 5. IDENTIFY MODEL FEATURES
# ============================================================

print("\n[4/12] Defining model features...")

metadata_columns = IDENTIFIER_COLUMNS | LABEL_METADATA_COLUMNS | EXCLUDED_MODEL_COLUMNS

candidate_features = [col for col in data.columns if col not in metadata_columns]

print("Candidate model features:", len(candidate_features))

# Convert only model features to numeric.
# Original metadata columns remain unchanged.
for col in candidate_features:

    data[col] = pd.to_numeric(data[col], errors="coerce")

data[candidate_features] = data[candidate_features].replace([np.inf, -np.inf], np.nan)


# ============================================================
# 6. REMOVE INVALID NUMERICAL RECORDS
# ============================================================

print("\n[5/12] Cleaning invalid numerical values...")

before = len(data)

data = data.dropna(subset=candidate_features).copy()

removed_nonfinite = before - len(data)

print("Nonfinite/missing records removed:", removed_nonfinite)

before = len(data)

for col in NONNEGATIVE_COLUMNS:

    if col in data.columns:

        data = data.loc[data[col] >= 0].copy()

removed_negative = before - len(data)

print("Invalid negative records removed:", removed_negative)


# ============================================================
# 7. REMOVE CONFLICTING LABELS
# ============================================================

print("\n[6/12] Removing conflicting-label patterns...")

# Keep this consistent with the original V3:
# duplicate detection is based on modelling features,
# not identifiers or source metadata.

hashes = pd.util.hash_pandas_object(data[candidate_features], index=False)

label_counts = data.groupby(hashes, sort=False)["Label"].nunique()

conflicting_hashes = label_counts[label_counts > 1].index

before = len(data)

data = data.loc[~hashes.isin(conflicting_hashes)].copy()

removed_conflicts = before - len(data)

print("Conflicting-label records removed:", removed_conflicts)


# ============================================================
# 8. REMOVE DUPLICATE FEATURE VECTORS
# ============================================================

print("\n[7/12] Removing duplicate feature vectors...")

before = len(data)

data = data.drop_duplicates(subset=candidate_features).copy()

removed_duplicates = before - len(data)

print("Duplicate feature vectors removed:", removed_duplicates)


# ============================================================
# 9. PRESERVE ALL ORIGINAL COLUMNS
# ============================================================

print("\n[8/12] Preserving full dataset schema...")

# Unlike the previous script, do not drop constant,
# redundant, identifier, or attempted-category columns.
# They remain available for auditing and traceability.

missing_original_columns = [col for col in raw_column_names if col not in data.columns]

if missing_original_columns:
    raise ValueError("Original columns were lost: " + str(missing_original_columns))

# Verify requested restored columns.
missing_requested = [
    col for col in EXPECTED_RESTORED_COLUMNS if col not in data.columns
]

if missing_requested:
    raise ValueError(
        "Requested columns missing from raw data: " + str(missing_requested)
    )

print("Original columns preserved:", len(raw_column_names))

print("Current columns:", len(data.columns))

# Remove unknown-family attempted flows from official experiment.
before = len(data)

data = data.loc[
    ~data["is_attempted"].eq(True) | ~data["Label"].isin(UNKNOWN_FAMILIES)
].copy()

removed_unknown_attempted = before - len(data)

print("Excluded attempted held-out-family records:", removed_unknown_attempted)

data = data.reset_index(drop=True)


# ============================================================
# 10. FINAL MODEL FEATURE LIST
# ============================================================

print("\n[9/12] Finalizing model feature contract...")

# Constant/redundant features remain in the full dataframe,
# but must not enter model training.
model_features = [
    col for col in candidate_features if data[col].nunique(dropna=False) > 1
]

assert "Label" not in model_features
assert "source_file" not in model_features
assert "Attempted Category" not in model_features
assert "is_attempted" not in model_features

print("Final model features:", len(model_features))
print("Full dataset columns:", len(data.columns))

print("\nCleaned class distribution:")
print(data["Label"].value_counts())


# ============================================================
# 11. KNOWN / UNKNOWN SPLITTING
# ============================================================

print("\n[10/12] Creating known/unknown experiment...")

available_labels = set(data["Label"].unique())

for attack in UNKNOWN_FAMILIES:

    if attack not in available_labels:
        raise ValueError(f"Unknown attack {attack} not found.")

unknown_df = data.loc[data["Label"].isin(UNKNOWN_FAMILIES)].copy()

known_df = data.loc[~data["Label"].isin(UNKNOWN_FAMILIES)].copy()

assert len(data) == len(known_df) + len(unknown_df)

assert not known_df["Label"].isin(UNKNOWN_FAMILIES).any()

known_train, temp = train_test_split(
    known_df, test_size=0.30, random_state=SEED, stratify=known_df["Label"]
)

known_validation, known_test = train_test_split(
    temp, test_size=0.50, random_state=SEED, stratify=temp["Label"]
)

ae_train = known_train.loc[known_train["Label"] == "BENIGN"].copy()

ae_validation = known_validation.loc[known_validation["Label"] == "BENIGN"].copy()

print("Known train:", known_train.shape)
print("Known validation:", known_validation.shape)
print("Known test:", known_test.shape)
print("Unknown test:", unknown_df.shape)

print("AE BENIGN train:", ae_train.shape)
print("AE BENIGN validation:", ae_validation.shape)


# ============================================================
# 12. FINAL VALIDATION
# ============================================================

print("\n[11/12] Validating outputs...")

assert ae_train["Label"].eq("BENIGN").all()
assert ae_validation["Label"].eq("BENIGN").all()

assert not known_train["Label"].isin(UNKNOWN_FAMILIES).any()

assert not known_validation["Label"].isin(UNKNOWN_FAMILIES).any()

assert not known_test["Label"].isin(UNKNOWN_FAMILIES).any()

assert set(known_train.index).isdisjoint(known_validation.index)

assert set(known_train.index).isdisjoint(known_test.index)

assert set(known_validation.index).isdisjoint(known_test.index)

assert data[model_features].isna().sum().sum() == 0

assert np.isfinite(data[model_features].to_numpy(dtype=np.float64)).all()

print("All validation checks passed.")


# ============================================================
# 13. SAVE ALL OUTPUTS
# ============================================================

print("\n[12/12] Saving datasets...")

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
