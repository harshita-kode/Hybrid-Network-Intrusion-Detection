from pathlib import Path
import pandas as pd
import json
import numpy as np

from sklearn.model_selection import train_test_split

# ============================================================
# GC2 - UNKNOWN ATTACK DATA SPLITTING
#
# Initial KB       : 60%
# Validation       : 20%
# Final Testing    : 15%
# Demo / KB Update : 5%
#
# Dataset: Improved CIC-IDS2017
# ============================================================

SEED = 42

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

while ROOT.name != "Hybrid-Network-Intrusion-Detection":

    if ROOT == ROOT.parent:
        raise RuntimeError("Project root not found")

    ROOT = ROOT.parent

DATA_DIR = ROOT / "data" / "processed" / "Data_V3_SSH_Unknown_final"

INPUT_FILE = DATA_DIR / "unknown_test.parquet"

OUT_DIR = DATA_DIR / "unknown_kb_splits"

OUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

print("Input:", INPUT_FILE)
print("Output:", OUT_DIR)


# ------------------------------------------------------------
# 2. LOAD UNKNOWN DATA
# ------------------------------------------------------------

unknown_df = pd.read_parquet(INPUT_FILE)

print("\nOriginal unknown dataset:", unknown_df.shape)

print("\nClass distribution:")
print(unknown_df["Label"].value_counts())

UNKNOWN_CLASSES = ["FTP-Patator", "SSH-Patator", "DoS Slowhttptest"]

assert set(unknown_df["Label"].unique()) == set(UNKNOWN_CLASSES)

assert len(unknown_df) == 8673


# ------------------------------------------------------------
# 3. INITIAL KB - 60%
# ------------------------------------------------------------

initial_kb, remaining = train_test_split(
    unknown_df, test_size=0.40, random_state=SEED, stratify=unknown_df["Label"]
)

print("\nInitial KB:", initial_kb.shape)
print("Remaining:", remaining.shape)


# ------------------------------------------------------------
# 4. VALIDATION - 20%
# ------------------------------------------------------------

# Remaining data = 40% of original.
# Validation = 20% of original = 50% of remaining.

validation, remaining = train_test_split(
    remaining, test_size=0.50, random_state=SEED, stratify=remaining["Label"]
)

print("\nValidation:", validation.shape)
print("Remaining:", remaining.shape)


# ------------------------------------------------------------
# 5. FINAL TESTING - 15%
# ------------------------------------------------------------

# Remaining = 20% of original.
# Final testing = 15% of original.
# Demo = 5% of original.
#
# Therefore final testing gets 75% of remaining.

final_test, demo_kb = train_test_split(
    remaining, test_size=0.25, random_state=SEED, stratify=remaining["Label"]
)

print("\nFinal test:", final_test.shape)
print("Demo / KB updates:", demo_kb.shape)


# ------------------------------------------------------------
# 6. RESET INDEX
# ------------------------------------------------------------

initial_kb = initial_kb.reset_index(drop=True)
validation = validation.reset_index(drop=True)
final_test = final_test.reset_index(drop=True)
demo_kb = demo_kb.reset_index(drop=True)


# ------------------------------------------------------------
# 7. VALIDATE SPLITS
# ------------------------------------------------------------

total = len(initial_kb) + len(validation) + len(final_test) + len(demo_kb)

assert total == len(unknown_df)

# Check original row identifiers are disjoint.
# Use source_file + id because id may repeat across CSV files.


def get_keys(df):
    return set(zip(df["source_file"].astype(str), df["id"]))


keys = [
    get_keys(initial_kb),
    get_keys(validation),
    get_keys(final_test),
    get_keys(demo_kb),
]

for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        assert keys[i].isdisjoint(keys[j]), f"Overlap between splits {i} and {j}"

# Verify every class appears in every split.

for frame in [initial_kb, validation, final_test, demo_kb]:
    assert set(frame["Label"].unique()) == set(UNKNOWN_CLASSES)

print("\nAll split validation checks passed!")


# ------------------------------------------------------------
# 8. SAVE DATASETS
# ------------------------------------------------------------

outputs = {
    "unknown_initial_kb": initial_kb,
    "unknown_validation": validation,
    "unknown_final_test": final_test,
    "unknown_demo_kb_updates": demo_kb,
}

for name, frame in outputs.items():

    parquet_path = OUT_DIR / f"{name}.parquet"

    frame.to_parquet(parquet_path, index=False)

    print(f"Saved {name}: " f"{len(frame):,} records")


# ------------------------------------------------------------
# 9. CLASS DISTRIBUTION REPORT
# ------------------------------------------------------------

summary = (
    pd.concat(
        {
            "Initial_KB_60%": initial_kb["Label"].value_counts(),
            "Validation_20%": validation["Label"].value_counts(),
            "Final_Test_15%": final_test["Label"].value_counts(),
            "Demo_KB_5%": demo_kb["Label"].value_counts(),
        },
        axis=1,
    )
    .fillna(0)
    .astype(int)
)

summary.loc["TOTAL"] = summary.sum()

summary.to_csv(OUT_DIR / "unknown_kb_split_summary.csv")

print("\nCLASS-WISE SPLIT SUMMARY:")
print(summary)


# ------------------------------------------------------------
# 10. SAVE CONFIGURATION
# ------------------------------------------------------------

config = {
    "dataset": "Improved CIC-IDS2017",
    "source": "unknown_test.parquet",
    "seed": SEED,
    "unknown_classes": UNKNOWN_CLASSES,
    "split_ratios": {
        "initial_kb": 0.60,
        "validation": 0.20,
        "final_test": 0.15,
        "demo_kb_updates": 0.05,
    },
    "total_unknown_records": len(unknown_df),
}

with open(OUT_DIR / "unknown_kb_config.json", "w") as f:
    json.dump(config, f, indent=2)


# ------------------------------------------------------------
# 11. SAVE TXT DOCUMENTATION
# ------------------------------------------------------------

LOG_FILE = OUT_DIR / "unknown_kb_split_log.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:

    f.write("GC2 UNKNOWN ATTACK DATA SPLITTING\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Source: {INPUT_FILE}\n")
    f.write(f"Total records: {len(unknown_df)}\n")
    f.write(f"Total columns: {len(unknown_df.columns)}\n\n")

    f.write("SPLIT CONFIGURATION\n")
    f.write("-" * 60 + "\n")

    for name, frame in outputs.items():

        f.write(
            f"{name}: {len(frame)} records "
            f"({len(frame) / len(unknown_df) * 100:.2f}%)\n"
        )

    f.write("\nCLASS DISTRIBUTION\n")
    f.write("-" * 60 + "\n")
    f.write(summary.to_string())

    f.write("\n\nALL COLUMNS\n")
    f.write("-" * 60 + "\n")

    for i, col in enumerate(unknown_df.columns, 1):
        f.write(f"{i}. {col}\n")

    f.write("\nVALIDATION\n")
    f.write("-" * 60 + "\n")
    f.write("All records allocated exactly once.\n")
    f.write("All three attack classes represented.\n")
    f.write("No overlap between split identifiers.\n")

print("\n======================================")
print("UNKNOWN KB SPLITTING COMPLETED")
print("======================================")

print("Total unknown records:", len(unknown_df))
print("Output:", OUT_DIR)
