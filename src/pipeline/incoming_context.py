
"""
GC2 incoming-flow chronological context.

Prepares Version20 Autoencoder behavioural features
using supplied historical network flows.

This is a stateless, file-based evaluation/inference helper.
It does not maintain history across separate program runs.
"""

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AE_DIR = ROOT / "notebooks" / "autoencoder"

if str(AE_DIR) not in sys.path:
    sys.path.insert(0, str(AE_DIR))

from version20_common import add_behavioural_features


class IncomingContext:

    def __init__(self, history):
        if not isinstance(history, pd.DataFrame):
            raise TypeError(
                "History must be a pandas DataFrame"
            )

        self.history = history.copy()
        self.history.columns = (
            self.history.columns.astype(str).str.strip()
        )

        if "Timestamp" not in self.history.columns:
            raise ValueError(
                "History requires Timestamp"
            )

        self.history["_gc2_timestamp"] = pd.to_datetime(
            self.history["Timestamp"],
            errors="raise"
        )

        if self.history["_gc2_timestamp"].isna().any():
            raise ValueError(
                "History contains missing timestamps"
            )

        # Preserve supplied arrival order for equal timestamps.
        self.history["_gc2_order"] = range(
            len(self.history)
        )

        self.history = self.history.sort_values(
            ["_gc2_timestamp", "_gc2_order"],
            kind="stable"
        ).reset_index(drop=True)

    def prepare(self, flow, earlier_batch_flows=None):
        """
        Prepare one incoming flow using strictly earlier
        historical records and optional earlier batch flows.

        earlier_batch_flows must contain only records known
        to have arrived before the target flow.
        """

        if isinstance(flow, dict):
            flow = pd.DataFrame([flow])

        if not isinstance(flow, pd.DataFrame) or len(flow) != 1:
            raise ValueError(
                "Exactly one incoming flow is required"
            )

        target = flow.copy()
        target.columns = target.columns.astype(str).str.strip()

        if "Timestamp" not in target.columns:
            raise ValueError(
                "Incoming flow requires Timestamp"
            )

        target_time = pd.to_datetime(
            target.iloc[0]["Timestamp"],
            errors="raise"
        )

        # Strictly earlier timestamps only.
        prior = self.history.loc[
            self.history["_gc2_timestamp"] < target_time
        ].drop(
            columns=["_gc2_timestamp", "_gc2_order"]
        )

        if earlier_batch_flows is not None:
            batch = earlier_batch_flows.copy()

            if not batch.empty:
                batch_times = pd.to_datetime(
                    batch["Timestamp"],
                    errors="raise"
                )

                batch = batch.loc[
                    batch_times <= target_time
                ]

                prior = pd.concat(
                    [prior, batch],
                    ignore_index=True
                )

        # Sort the full prior context chronologically.
        if not prior.empty:
            prior = prior.copy()
            prior["_gc2_timestamp"] = pd.to_datetime(
                prior["Timestamp"],
                errors="raise"
            )

            prior["_gc2_order"] = range(len(prior))

            prior = prior.sort_values(
                ["_gc2_timestamp", "_gc2_order"],
                kind="stable"
            ).drop(
                columns=["_gc2_timestamp", "_gc2_order"]
            )

        # Target flow is appended last.
        context = pd.concat(
            [prior, target],
            ignore_index=True
        )

        prepared = add_behavioural_features(
            context
        ).iloc[-1].to_dict()

        return prepared, len(prior)
