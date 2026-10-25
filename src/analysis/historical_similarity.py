
"""
Historical attack-profile similarity.

Compares one incoming network flow against labelled
historical attack references.

Similarity is supporting evidence, not a classifier.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]

PROFILE_DIR = (
    ROOT / "data/processed/historical_profiles"
)


class HistoricalSimilarity:

    def __init__(self, profile_dir=PROFILE_DIR):

        profile_dir = Path(profile_dir)

        self.profiles = pd.read_parquet(
            profile_dir / "attack_profiles.parquet"
        )

        self.features = joblib.load(
            profile_dir / "comparison_features.joblib"
        )

        self.labels = (
            self.profiles["Label"].astype(str).to_numpy()
        )

        X = self.profiles[
            self.features
        ].to_numpy(dtype=np.float64)

        self.scaler = RobustScaler()

        X_scaled = self.scaler.fit_transform(X)

        self.neighbors = NearestNeighbors(
            n_neighbors=1,
            metric="euclidean"
        )

        self.neighbors.fit(X_scaled)

        print(
            "Historical attack profiles loaded:",
            len(self.profiles)
        )

    def compare(self, flow):

        if isinstance(flow, dict):
            flow = pd.DataFrame([flow])

        if not isinstance(flow, pd.DataFrame):
            raise TypeError(
                "Flow must be a DataFrame or dictionary"
            )

        if len(flow) != 1:
            raise ValueError(
                "Exactly one flow must be supplied"
            )

        missing = [
            feature
            for feature in self.features
            if feature not in flow.columns
        ]

        if missing:
            raise ValueError(
                f"Missing comparison features: {missing}"
            )

        X = flow[
            self.features
        ].to_numpy(dtype=np.float64)

        if not np.isfinite(X).all():
            raise ValueError(
                "Flow contains nonfinite comparison features"
            )

        X_scaled = self.scaler.transform(X)

        distances, indices = self.neighbors.kneighbors(
            X_scaled,
            n_neighbors=1
        )

        nearest_index = int(indices[0, 0])
        nearest_distance = float(distances[0, 0])

        nearest_label = str(
            self.labels[nearest_index]
        )

        return {
            "nearest_historical_attack": nearest_label,
            "distance": nearest_distance,
            "method": "robust_scaled_euclidean",
            "reference_count": len(self.profiles),
            "interpretation": (
                "Nearest labelled historical attack profile. "
                "Distance is not a calibrated probability "
                "or confirmation of attack identity."
            ),
        }
