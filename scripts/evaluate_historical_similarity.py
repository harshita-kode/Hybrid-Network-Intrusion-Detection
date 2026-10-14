
"""
GC2 historical attack-profile evaluation.

Evaluates:
1. Overall same-class nearest-profile rate.
2. Per-class matching rates.
3. Correct/mismatched nearest-neighbour distances.
4. Benign-to-attack-reference distances.
5. Held-out unknown-to-attack-reference distances.

This evaluates historical similarity as supporting
analyst evidence, not as a deployed classifier.

No model training or KB updates are performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.historical_similarity import HistoricalSimilarity


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT / "data"
    / "processed"
    / "Data_V3_SSH_Unknown_final"
)

OUTPUT_DIR = ROOT / "outputs" / "results" / "historical_similarity"

KNOWN_TEST = DATA_DIR / "known_test.parquet"
UNKNOWN_TEST = DATA_DIR / "unknown_test.parquet"

RANDOM_STATE = 42
MAX_KNOWN_PER_CLASS = 100
MAX_BENIGN = 500
MAX_UNKNOWN = 500


def sample_group(group, maximum):
    return group.sample(
        n=min(maximum, len(group)),
        random_state=RANDOM_STATE
    )


def prepare_numeric(data, features):
    missing = [
        feature for feature in features
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset missing comparison features: {missing}"
        )

    X = data[features].apply(
        pd.to_numeric,
        errors="raise"
    )

    values = X.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            "Nonfinite comparison features encountered"
        )

    return X


def compare_batch(comparator, data):
    """
    Calculate nearest historical attack profiles
    in one batch using the already-fitted scaler
    and nearest-neighbour index.
    """

    X = prepare_numeric(data, comparator.features)

    scaled = comparator.scaler.transform(
        X.to_numpy(dtype=np.float64)
    )

    distances, indices = comparator.neighbors.kneighbors(
        scaled,
        n_neighbors=1
    )

    nearest_indices = indices[:, 0]

    return pd.DataFrame({
        "nearest_historical_attack": (
            comparator.labels[nearest_indices]
        ),
        "distance": distances[:, 0],
    })


def evaluate_known(comparator, known):
    attacks = known.loc[
        known["Label"].astype(str).str.upper() != "BENIGN"
    ].copy()

    sampled = pd.concat(
        [
            sample_group(group, MAX_KNOWN_PER_CLASS)
            for _, group in attacks.groupby("Label")
        ],
        ignore_index=True
    )

    results = compare_batch(comparator, sampled)

    results.insert(
        0,
        "actual_class",
        sampled["Label"].astype(str).to_numpy()
    )

    results["same_class"] = (
        results["actual_class"]
        == results["nearest_historical_attack"]
    )

    overall = float(
        results["same_class"].mean()
    )

    per_class = (
        results.groupby("actual_class")
        .agg(
            samples=("same_class", "size"),
            same_class_rate=("same_class", "mean"),
            mean_distance=("distance", "mean"),
            median_distance=("distance", "median"),
        )
        .reset_index()
    )

    return results, per_class, overall


def distance_summary(results, group_name):
    if results.empty:
        return None

    return {
        "group": group_name,
        "samples": len(results),
        "mean_distance": float(results["distance"].mean()),
        "median_distance": float(results["distance"].median()),
        "p90_distance": float(
            results["distance"].quantile(0.90)
        ),
        "min_distance": float(results["distance"].min()),
        "max_distance": float(results["distance"].max()),
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading historical comparison model...")
    comparator = HistoricalSimilarity()

    print("Loading known-test dataset...")
    known = pd.read_parquet(KNOWN_TEST)

    # -----------------------------------------------
    # KNOWN ATTACKS
    # -----------------------------------------------

    print("\nEvaluating known attacks...")

    known_results, per_class, overall = evaluate_known(
        comparator,
        known
    )

    known_results.to_csv(
        OUTPUT_DIR / "known_attack_results.csv",
        index=False
    )

    per_class.to_csv(
        OUTPUT_DIR / "per_class_results.csv",
        index=False
    )

    print("\n========== KNOWN ATTACK EVALUATION ==========")
    print("Samples tested:", len(known_results))
    print(f"Overall same-class rate: {overall:.2%}")

    print("\nPer-class results:")
    print(per_class.to_string(index=False))

    # -----------------------------------------------
    # CORRECT VS INCORRECT NEAREST PROFILES
    # -----------------------------------------------

    correct = known_results.loc[
        known_results["same_class"]
    ]

    incorrect = known_results.loc[
        ~known_results["same_class"]
    ]

    # -----------------------------------------------
    # BENIGN TRAFFIC
    # -----------------------------------------------

    print("\nEvaluating benign traffic...")

    benign = known.loc[
        known["Label"].astype(str).str.upper() == "BENIGN"
    ]

    benign = sample_group(
        benign,
        MAX_BENIGN
    )

    benign_results = compare_batch(
        comparator,
        benign
    )

    benign_results.to_csv(
        OUTPUT_DIR / "benign_results.csv",
        index=False
    )

    # -----------------------------------------------
    # HELD-OUT UNKNOWN TRAFFIC
    # -----------------------------------------------

    unknown_results = None

    if UNKNOWN_TEST.is_file():
        print("\nEvaluating held-out unknown traffic...")

        unknown = pd.read_parquet(
            UNKNOWN_TEST
        )

        unknown = sample_group(
            unknown,
            MAX_UNKNOWN
        )

        unknown_results = compare_batch(
            comparator,
            unknown
        )

        if "Label" in unknown.columns:
            unknown_results.insert(
                0,
                "actual_class",
                unknown["Label"].astype(str).to_numpy()
            )

        unknown_results.to_csv(
            OUTPUT_DIR / "unknown_results.csv",
            index=False
        )

    # -----------------------------------------------
    # DISTANCE DISTRIBUTIONS
    # -----------------------------------------------

    summaries = [
        distance_summary(
            correct,
            "known_same_class"
        ),
        distance_summary(
            incorrect,
            "known_different_class"
        ),
        distance_summary(
            benign_results,
            "benign"
        ),
    ]

    if unknown_results is not None:
        summaries.append(
            distance_summary(
                unknown_results,
                "held_out_unknown"
            )
        )

    summaries = [
        item for item in summaries
        if item is not None
    ]

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        OUTPUT_DIR / "distance_summary.csv",
        index=False
    )

    print("\n========== DISTANCE SUMMARY ==========")
    print(summary_df.to_string(index=False))

    print("\nSaved evaluation files to:")
    print(OUTPUT_DIR)

    print("\nHISTORICAL SIMILARITY EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
