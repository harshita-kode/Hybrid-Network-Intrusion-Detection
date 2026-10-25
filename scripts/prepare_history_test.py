
"""Prepare reproducible incoming-history test files."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "Data_V3_SSH_Unknown_final"
    / "known_test.parquet"
)

OUTPUT = ROOT / "data" / "incoming"

TEST_ROW = 235964


def main():

    print("Loading known-test dataset...")

    data = pd.read_parquet(SOURCE)

    times = pd.to_datetime(
        data["Timestamp"],
        errors="raise"
    )

    target_time = times.iloc[TEST_ROW]

    positions = np.arange(len(data))

    # Include only records that precede the target
    # by timestamp and original row position.
    prior_mask = (
        (times < target_time)
        |
        (
            (times == target_time)
            &
            (positions < TEST_ROW)
        )
    )

    history = data.loc[prior_mask].copy()

    target = data.iloc[[TEST_ROW]].copy()

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    history.to_parquet(
        OUTPUT / "gc2_history_test.parquet",
        index=False
    )

    target.to_csv(
        OUTPUT / "gc2_incoming_test.csv",
        index=False
    )

    print("Historical flows:", len(history))
    print("Target row:", TEST_ROW)
    print("Saved test history and incoming flow.")


if __name__ == "__main__":
    main()
