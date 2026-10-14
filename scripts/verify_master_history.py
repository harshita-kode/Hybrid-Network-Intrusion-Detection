
"""
Verify that known/unknown evaluation records can be
located in the original clean_master dataset.

Diagnostic only. Does not modify any data or models.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT / "data/processed/Data_V3_SSH_Unknown_final"
)

MANIFEST_PATH = (
    ROOT / "outputs/results/gc2_evaluation"
    / "unknown_detection/evaluation_manifest.csv"
)

KEY_COLUMNS = [
    "Flow ID",
    "Timestamp",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
]


def main():

    print("Loading evaluation manifest...")

    manifest = pd.read_csv(MANIFEST_PATH)

    print("Loading master flow identifiers...")

    master = pd.read_parquet(
        DATA_DIR / "clean_master.parquet",
        columns=KEY_COLUMNS
    )

    master_keys = pd.util.hash_pandas_object(
        master[KEY_COLUMNS],
        index=False
    )

    master_counts = master_keys.value_counts()

    results = []

    for split in manifest["source_split"].unique():

        print("\nChecking:", split)

        source = pd.read_parquet(
            DATA_DIR / f"{split}.parquet",
            columns=KEY_COLUMNS
        )

        selected = manifest.loc[
            manifest["source_split"] == split
        ]

        positions = selected[
            "source_row"
        ].astype(int).to_numpy()

        target = source.iloc[positions]

        target_keys = pd.util.hash_pandas_object(
            target[KEY_COLUMNS],
            index=False
        )

        counts = target_keys.map(
            master_counts
        ).fillna(0).astype(int)

        for position, count in zip(
            positions,
            counts
        ):
            results.append({
                "source_split": split,
                "source_row": int(position),
                "master_match_count": int(count),
            })

    result = pd.DataFrame(results)

    print("\n========== MASTER HISTORY CHECK ==========")

    print("Selected flows:", len(result))

    print(
        "Exactly one master match:",
        int((result["master_match_count"] == 1).sum())
    )

    print(
        "Missing from master:",
        int((result["master_match_count"] == 0).sum())
    )

    print(
        "Multiple master matches:",
        int((result["master_match_count"] > 1).sum())
    )

    output = (
        ROOT / "outputs/results/gc2_evaluation"
        / "unknown_detection/master_history_check.csv"
    )

    result.to_csv(
        output,
        index=False
    )

    print("\nSaved:", output)

    if not (result["master_match_count"] == 1).all():
        print(
            "\nWARNING: Some records need further "
            "identity verification."
        )
    else:
        print(
            "\nMASTER RECORD IDENTITY CHECK: PASSED"
        )


if __name__ == "__main__":
    main()
