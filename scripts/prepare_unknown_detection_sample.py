
"""Prepare a reproducible GC2 unknown-detection evaluation sample."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT / "data/processed/Data_V3_SSH_Unknown_final"
)

ROUTING_PATH = (
    ROOT / "outputs/results/gc2_evaluation"
    / "unknown_routing/unknown_predictions.csv"
)

OUTPUT_DIR = (
    ROOT / "outputs/results/gc2_evaluation"
    / "unknown_detection"
)

RANDOM_STATE = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")

    unknown = pd.read_parquet(
        DATA_DIR / "unknown_test.parquet"
    )

    known = pd.read_parquet(
        DATA_DIR / "known_test.parquet"
    )

    routing = pd.read_csv(ROUTING_PATH)

    if len(routing) != len(unknown):
        raise ValueError("Unknown routing row count mismatch")

    expected = unknown["Label"].astype(str).to_numpy()

    if not (
        routing["actual_unknown_class"]
        .astype(str)
        .to_numpy() == expected
    ).all():
        raise ValueError(
            "Routing results do not align with unknown-test rows"
        )

    selected = []

    for attack in sorted(unknown["Label"].unique()):

        candidates = routing.loc[
            (routing["actual_unknown_class"] == attack)
            & routing["accepted_as_benign"].astype(bool)
        ]

        sample = candidates.sample(
            n=min(30, len(candidates)),
            random_state=RANDOM_STATE
        )

        selected.append(
            pd.DataFrame({
                "source_split": "unknown_test",
                "source_row": sample["test_row"].astype(int),
                "actual_label": attack,
                "selection_group": "high_confidence_benign",
            })
        )

    benign_positions = known.index[
        known["Label"].astype(str) == "BENIGN"
    ]

    benign_sample = pd.Series(
        benign_positions
    ).sample(
        n=min(90, len(benign_positions)),
        random_state=RANDOM_STATE
    )

    selected.append(
        pd.DataFrame({
            "source_split": "known_test",
            "source_row": benign_sample.astype(int).to_numpy(),
            "actual_label": "BENIGN",
            "selection_group": "benign_control",
        })
    )

    manifest = pd.concat(
        selected,
        ignore_index=True
    )

    manifest.to_csv(
        OUTPUT_DIR / "evaluation_manifest.csv",
        index=False
    )

    print("\n========== EVALUATION SAMPLE ==========")
    print(
        manifest.groupby(
            ["actual_label", "selection_group"]
        ).size().to_string()
    )

    print("\nTotal selected:", len(manifest))
    print("Saved:", OUTPUT_DIR / "evaluation_manifest.csv")


if __name__ == "__main__":
    main()
