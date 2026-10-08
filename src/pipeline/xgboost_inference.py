import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from src.models.confidence import (
    route_by_confidence,
    DEFAULT_CONFIDENCE_THRESHOLD
)


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
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
    "Data_V2",
    "known_train.parquet"
)


def load_model():

    model = joblib.load(MODEL_PATH)

    print(f"Model loaded from: {MODEL_PATH}")

    return model


def load_label_encoder():

    train_labels = pd.read_parquet(
        TRAIN_PATH,
        columns=["Label"]
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

    probability_dicts = []

    for row in probabilities:

        probability_dict = {
            class_name: float(probability)
            for class_name, probability
            in zip(
                label_encoder.classes_,
                row
            )
        }

        probability_dicts.append(
            probability_dict
        )

    results = pd.DataFrame({

        "predicted_class": predictions,

        "confidence": confidence,

        "status": status,

        "probabilities": probability_dicts

    })

    return results