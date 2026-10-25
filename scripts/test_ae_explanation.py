
"""Test Version20 explanation on a KB-unmatched demo flow."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notebooks.kb.kb_flow import (
    load_version20,
    prepare_ae_context,
    score_precomputed_flow,
)

from src.analysis.anomaly_explanation import (
    explain_autoencoder,
)


DATA_DIR = (
    ROOT
    / "data"
    / "processed"
    / "Data_V3_SSH_Unknown_final"
)

DEMO_PATH = (
    DATA_DIR
    / "unknown_kb_splits"
    / "unknown_demo_kb_updates.parquet"
)

AUDIT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "knn_kb"
    / "demo_similarity_audit"
    / "demo_similarity_audit.csv"
)

DEMO_ROW = 79


def main():

    print("Loading KB similarity audit...")

    audit = pd.read_csv(AUDIT_PATH)

    audit_match = audit.loc[
        audit["demo_row"] == DEMO_ROW
    ]

    if len(audit_match) != 1:
        raise ValueError(
            "Demo row not uniquely found in audit"
        )

    matched = str(
        audit_match.iloc[0]["matched"]
    ).strip().lower() == "true"

    if matched:
        raise ValueError(
            "Selected demo row is KB-matched; "
            "Autoencoder should not be called"
        )

    print("Loading original demonstration flow...")

    demo = pd.read_parquet(DEMO_PATH)

    if DEMO_ROW >= len(demo):
        raise IndexError("Demo row is out of range")

    flow = demo.iloc[DEMO_ROW].to_dict()

    # demo_row is audit metadata, not an AE feature.
    flow.pop("demo_row", None)

    print("Preparing chronological behavioural context...")

    # Do not pass DEMO_ROW as source_row_id.
    # It is the demo-partition position, not necessarily
    # the original unknown_test.parquet row index.
    prepared_flow, history_count = prepare_ae_context(
        ROOT,
        flow
    )

    print("Loading Version20 Autoencoder ensemble...")

    loaded_ae = load_version20(ROOT)

    print("Calculating feature-level explanations...")

    result = explain_autoencoder(
        flow=prepared_flow,
        loaded_ae=loaded_ae,
        top_n=5
    )

    # Independently compare with the KB workflow scorer.
    original_score, original_threshold = (
        score_precomputed_flow(
            prepared_flow,
            loaded_ae
        )
    )

    assert np.isclose(
        result["anomaly_mse"],
        original_score,
        rtol=1e-5,
        atol=1e-7
    )

    assert np.isclose(
        result["threshold"],
        original_threshold
    )

    print("\n========== AUTOENCODER EXPLANATION ==========")

    print("Demo row:", DEMO_ROW)
    print("KB route: AUTOENCODER")
    print("Historical flows:", history_count)
    print("Features:", result["feature_count"])
    print("Ensemble models:", result["ensemble_size"])

    print(
        f"Reconstruction MSE: "
        f"{result['anomaly_mse']:.8f}"
    )

    print(
        f"Anomaly threshold: "
        f"{result['threshold']:.8f}"
    )

    print("Anomalous:", result["anomalous"])

    print("\nTop reconstruction-error features:")

    for rank, item in enumerate(
        result["top_features"],
        start=1
    ):
        print(
            f"{rank}. {item['feature']}"
        )

        print(
            f"   Squared error: "
            f"{item['squared_error']:.8f}"
        )

        print(
            f"   Error share: "
            f"{item['error_share']:.2%}"
        )

    print("\nMSE consistency: PASSED")
    print("=============================================")


if __name__ == "__main__":
    main()
