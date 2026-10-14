
"""
Prepare Version20 chronological context for known-test flows.

This helper is for reproducible Data V3 integration testing.
It is not a live-traffic history manager.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AE_DIR = ROOT / "notebooks" / "autoencoder"

if str(AE_DIR) not in sys.path:
    sys.path.insert(0, str(AE_DIR))

from version20_common import add_behavioural_features


KNOWN_TEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "Data_V3_SSH_Unknown_final"
    / "known_test.parquet"
)


class KnownTestContext:

    def __init__(self):

        print("Loading known-test chronological history...")

        self.data = pd.read_parquet(KNOWN_TEST_PATH)

        self.data.columns = (
            self.data.columns.astype(str).str.strip()
        )

        self.times = pd.to_datetime(
            self.data["Timestamp"],
            errors="coerce"
        )

        if self.times.isna().any():
            raise ValueError(
                "Invalid timestamps in known-test history"
            )

        self.positions = np.arange(len(self.data))

    def prepare(self, row_index):
        """
        Reproduce the original benign-audit context.

        row_index must be the original positional index
        in known_test.parquet.
        """

        if type(row_index) is not int:
            raise TypeError("row_index must be an integer")

        if not 0 <= row_index < len(self.data):
            raise IndexError("Known-test row is out of range")

        timestamp = self.times.iloc[row_index]

        prior = (
            (self.times < timestamp)
            |
            (
                (self.times == timestamp)
                &
                (self.positions < row_index)
            )
        )

        context = pd.concat(
            [
                self.data.loc[prior],
                self.data.iloc[[row_index]]
            ],
            ignore_index=True
        )

        prepared = add_behavioural_features(
            context
        ).iloc[-1].to_dict()

        return prepared, int(prior.sum())
