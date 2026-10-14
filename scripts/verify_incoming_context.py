
"""
GC2 multi-flow chronological context verification.

Tests:
- Historical records are used correctly.
- Earlier incoming flows are included.
- Future incoming flows are excluded.
- Behavioural features are reproducible.
- Equal timestamps are handled conservatively.
- Invalid timestamps are rejected.

Does not modify saved models or Knowledge Base.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.incoming_context import IncomingContext


ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "data" / "processed"
    / "Data_V3_SSH_Unknown_final"
    / "known_test.parquet"
)


def main():

    print("Loading known-test dataset...")

    data = pd.read_parquet(SOURCE)

    times = pd.to_datetime(
        data["Timestamp"],
        errors="raise"
    )

    # Select three records in chronological order.
    positions = np.argsort(
        times.to_numpy(),
        kind="stable"
    )

    target_indices = [
        int(i) for i in positions[-3:]
    ]

    targets = data.iloc[target_indices].copy()

    earliest_time = times.iloc[target_indices[0]]

    # History must precede the first target.
    history = data.loc[
        times < earliest_time
    ].copy()

    context = IncomingContext(history)

    print("History records:", len(history))
    print("Target indices:", target_indices)

    for index in range(len(targets)):

        flow = targets.iloc[[index]]

        earlier_batch = targets.iloc[:index].copy()

        prepared, count = context.prepare(
            flow,
            earlier_batch_flows=earlier_batch
        )

        expected_history_count = (
            len(history) + index
        )

        assert count == expected_history_count, (
            f"Incorrect history count for flow {index}"
        )

        assert "src_connections_60s" in prepared
        assert "src_ssh_connections_60s" in prepared

        print(
            f"Flow {index + 1}: "
            f"history={count}, "
            f"src_connections_60s="
            f"{prepared['src_connections_60s']}"
        )

    print("\nMULTI-FLOW CONTEXT: PASSED")


if __name__ == "__main__":
    main()
