
"""Prepare a multi-flow incoming-history test."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "data/processed"
    / "Data_V3_SSH_Unknown_final"
    / "known_test.parquet"
)

OUTPUT = ROOT / "data/incoming"

# Real low-confidence rows identified earlier.
TEST_ROWS = [235964, 45333, 139216]


def main():

    print("Loading known-test dataset...")

    data = pd.read_parquet(SOURCE)

    times = pd.to_datetime(
        data["Timestamp"],
        errors="raise"
    )

    # Sort the selected records chronologically.
    rows = sorted(
        TEST_ROWS,
        key=lambda i: (times.iloc[i], i)
    )

    first_time = times.iloc[rows[0]]

    history_mask = times < first_time

    history = data.loc[history_mask].copy()

    incoming = data.iloc[rows].copy()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    history.to_parquet(
        OUTPUT / "gc2_multiflow_history.parquet",
        index=False
    )

    incoming.to_csv(
        OUTPUT / "gc2_multiflow_incoming.csv",
        index=False
    )

    print("Selected rows:", rows)
    print("History records:", len(history))
    print("Incoming flows:", len(incoming))

    print("Saved multi-flow test files.")


if __name__ == "__main__":
    main()
