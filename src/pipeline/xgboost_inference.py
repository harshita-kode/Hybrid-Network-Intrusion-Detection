import os
import sys

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

sys.path.insert(
    0,
    os.path.join(PROJECT_ROOT, "src")
)

from models.confidence import (
    route_by_confidence,
    DEFAULT_CONFIDENCE_THRESHOLD
)


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved-models",
    "xgboost_final.joblib"
)

TRAIN_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "classifier_train.csv"
)


def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    print(
        f"Model loaded from: {MODEL_PATH}"
    )

    return model


def load_label_encoder():

    train_labels = pd.read_csv(
        TRAIN_PATH,
        usecols=["Label"]
    )

    label_encoder = LabelEncoder()

    label_encoder.fit(
        train_labels["Label"]
    )

    return label_encoder


def prepare_features(data):

    columns_to_drop = []

    if "Label" in data.columns:
        columns_to_drop.append("Label")

    if "source_file" in data.columns:
        columns_to_drop.append("source_file")

    return data.drop(
        columns=columns_to_drop
    )


def predict_traffic(data):

    model = load_model()

    label_encoder = load_label_encoder()

    features = prepare_features(data)

    probabilities = model.predict_proba(
        features
    )

    predicted_indices = np.argmax(
        probabilities,
        axis=1
    )

    predictions = label_encoder.inverse_transform(
        predicted_indices
    )

    confidence, status = route_by_confidence(
        predictions,
        probabilities,
        threshold=DEFAULT_CONFIDENCE_THRESHOLD
    )

    results = pd.DataFrame({
        "predicted_class": predictions,
        "confidence": confidence,
        "status": status
    })

    return results


if __name__ == "__main__":

    print(
        "XGBoost inference module loaded successfully."
    )

    print(
        f"Confidence threshold: "
        f"{DEFAULT_CONFIDENCE_THRESHOLD}"
    )