
"""GC2 Stage 12: verify evaluation datasets."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT / "data/processed/Data_V3_SSH_Unknown_final"
)

SPLITS = DATA_DIR / "unknown_kb_splits"

KB_REFERENCE = (
    ROOT / "data/processed/knn_kb/knowledge_base.csv"
)


def keys(frame):
    return set(
        zip(
            frame["source_file"].astype(str),
            frame["id"].astype(str)
        )
    )


def main():

    print("\n========== STAGE 12 DATA VERIFICATION ==========")

    initial = pd.read_parquet(
        SPLITS / "unknown_initial_kb.parquet"
    )

    final_test = pd.read_parquet(
        SPLITS / "unknown_final_test.parquet"
    )

    demo = pd.read_parquet(
        SPLITS / "unknown_demo_kb_updates.parquet"
    )

    known = pd.read_parquet(
        DATA_DIR / "known_test.parquet"
    )

    reference = pd.read_csv(
        KB_REFERENCE,
        low_memory=False
    )

    benign = known.loc[
        known["Label"] == "BENIGN"
    ]

    print("Initial KB records:", len(initial))
    print("Final unknown-test records:", len(final_test))
    print("Demo records:", len(demo))
    print("Available benign controls:", len(benign))

    print("\nFinal-test attack distribution:")
    print(final_test["Label"].value_counts().to_string())

    initial_keys = keys(initial)
    final_keys = keys(final_test)
    demo_keys = keys(demo)
    reference_keys = keys(reference)

    if len(initial_keys) != len(initial):
        raise ValueError("Duplicate initial KB identifiers")

    if len(final_keys) != len(final_test):
        raise ValueError("Duplicate final-test identifiers")

    if initial_keys & final_keys:
        raise ValueError(
            "Initial KB overlaps final test"
        )

    if demo_keys & final_keys:
        raise ValueError(
            "Demo data overlaps final test"
        )

    if reference_keys != initial_keys:
        raise ValueError(
            "Frozen KB reference does not match initial split"
        )

    print("\nFrozen KB membership: VERIFIED")
    print("Final-test identity overlap: NONE")

    print("\nSTAGE 12 DATA VERIFICATION: PASSED")


if __name__ == "__main__":
    main()
