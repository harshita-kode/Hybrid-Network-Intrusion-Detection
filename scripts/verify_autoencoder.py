
"""Verify Autoencoder explanations on real BENIGN traffic."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AE_DIR = ROOT / "notebooks" / "autoencoder"
sys.path.insert(0, str(AE_DIR))

from version20_common import add_behavioural_features

from notebooks.kb.kb_flow import (
    load_version20,
    score_precomputed_flow,
)

from src.analysis.anomaly_explanation import (
    explain_autoencoder,
)


KNOWN_PATH = (
    ROOT / "data" / "processed"
    / "Data_V3_SSH_Unknown_final"
    / "known_test.parquet"
)

AUDIT_PATH = (
    ROOT / "data" / "processed"
    / "knn_kb" / "demo_similarity_audit"
    / "benign_demo_results.csv"
)

TEST_ROWS = [211335, 229252]


def verify_benign_result(result, audit_row):
    """Check explanation against the original benign audit."""

    assert result["feature_count"] == 93
    assert result["ensemble_size"] == 3

    assert np.isclose(
        result["anomaly_mse"],
        result["original_mse"],
        rtol=1e-5,
        atol=1e-7
    )

    assert np.isclose(
        result["anomaly_mse"],
        float(audit_row["ae_mse"]),
        rtol=1e-4,
        atol=1e-5
    ), (
        "Reconstruction MSE differs from saved benign audit"
    )

    assert np.isclose(
        result["threshold"],
        float(audit_row["ae_threshold"]),
        rtol=1e-5,
        atol=1e-7
    )

    audit_anomalous = (
        str(audit_row["ae_anomalous"])
        .strip().lower() == "true"
    )

    assert result["anomalous"] == audit_anomalous

    assert all(
        item["squared_error"] >= 0
        for item in result["top_features"]
    )

    assert all(
        0 <= item["error_share"] <= 1
        for item in result["top_features"]
    )


def main():

    print("Loading known-test dataset...")

    known = pd.read_parquet(KNOWN_PATH)
    known.columns = known.columns.astype(str).str.strip()

    print("Loading benign audit...")

    audit = pd.read_csv(AUDIT_PATH)

    times = pd.to_datetime(
        known["Timestamp"],
        errors="coerce"
    )

    if times.isna().any():
        raise ValueError(
            "Invalid timestamps in known-test dataset"
        )

    print("Loading Version20 Autoencoder ensemble...")

    loaded_ae = load_version20(ROOT)

    for row_index in TEST_ROWS:

        print(
            f"\n========== BENIGN ROW {row_index} =========="
        )

        if not 0 <= row_index < len(known):
            raise IndexError("Invalid known-test row")

        audit_match = audit.loc[
            audit["known_test_row"] == row_index
        ]

        if len(audit_match) != 1:
            raise ValueError(
                "Benign row not uniquely found in audit"
            )

        audit_row = audit_match.iloc[0]

        if bool(audit_row["kb_matched"]):
            raise ValueError(
                "Selected benign flow was KB-matched"
            )

        if not bool(audit_row["ae_called"]):
            raise ValueError(
                "Autoencoder was not called for this row"
            )

        flow = known.iloc[row_index]

        if str(flow["Label"]).strip().upper() != "BENIGN":
            raise ValueError(
                "Selected row is not BENIGN"
            )

        # --------------------------------------------------
        # REPRODUCE ORIGINAL CHRONOLOGICAL CONTEXT
        # --------------------------------------------------

        prior = (
            (times < times.iloc[row_index])
            |
            (
                (times == times.iloc[row_index])
                &
                (np.arange(len(known)) < row_index)
            )
        )

        context = pd.concat(
            [
                known.loc[prior],
                known.iloc[[row_index]]
            ],
            ignore_index=True
        )

        prepared = add_behavioural_features(
            context
        ).iloc[-1].to_dict()

        # --------------------------------------------------
        # EXPLAIN AUTOENCODER RESULT
        # --------------------------------------------------

        result = explain_autoencoder(
            flow=prepared,
            loaded_ae=loaded_ae,
            top_n=5
        )

        # Compare with original scoring function.
        original_mse, original_threshold = (
            score_precomputed_flow(
                prepared,
                loaded_ae
            )
        )

        assert np.isclose(
            result["anomaly_mse"],
            original_mse,
            rtol=1e-5,
            atol=1e-7
        )

        assert np.isclose(
            result["threshold"],
            original_threshold
        )

        verify_benign_result(
            result,
            audit_row
        )

        # --------------------------------------------------
        # DISPLAY
        # --------------------------------------------------

        print("Actual label: BENIGN")
        print("Historical flows:", int(prior.sum()))

        print(
            "Reconstruction MSE:",
            f"{result['anomaly_mse']:.8f}"
        )

        print(
            "Threshold:",
            f"{result['threshold']:.8f}"
        )

        print("Anomalous:", result["anomalous"])
        print("Saved outcome:", audit_row["outcome"])

        print("\nTop reconstruction-error features:")

        for item in result["top_features"]:
            print(
                f"  {item['feature']}: "
                f"{item['squared_error']:.6f} "
                f"({item['error_share']:.2%})"
            )

        print("\nBENIGN VERIFICATION: PASSED")

    print(
        "\nALL AUTOENCODER BENIGN VERIFICATION TESTS PASSED"
    )


if __name__ == "__main__":
    main()
