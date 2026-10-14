
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Data_V3_SSH_Unknown_final"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "saved-models"
    / "xgboost_final.joblib"
)


def load_shap_inputs():
    """
    Load the trained XGBoost model and matching
    Data V3 test features.

    Stops if the model and dataset are incompatible.
    """

    print("Loading trained XGBoost model...")

    saved_model = joblib.load(MODEL_PATH)

    # Handle both supported model formats.
    if hasattr(saved_model, "get_booster"):
        model = saved_model

    elif hasattr(saved_model, "model"):
        model = saved_model.model

    else:
        raise TypeError(
            "Unsupported saved XGBoost model type."
        )

    booster = model.get_booster()

    # --------------------------------------------------
    # LOAD FEATURE CONFIGURATION
    # --------------------------------------------------

    feature_path = DATA_DIR / "feature_list.json"

    with open(feature_path, encoding="utf-8") as file:
        feature_config = json.load(file)

    # Support a plain feature list or a dictionary
    # containing a feature list.
    if isinstance(feature_config, list):
        feature_names = feature_config

    elif isinstance(feature_config, dict):
        feature_names = feature_config.get("features")

        if feature_names is None:
            feature_names = feature_config.get(
                "feature_names"
            )

    else:
        feature_names = None

    if (
        not isinstance(feature_names, list)
        or not feature_names
    ):
        raise ValueError(
            "Could not read model features from "
            "feature_list.json."
        )

    if len(set(feature_names)) != len(feature_names):
        raise ValueError(
            "Duplicate feature names in feature_list.json."
        )

    # --------------------------------------------------
    # VERIFY MODEL COMPATIBILITY
    # --------------------------------------------------

    model_feature_count = booster.num_features()

    if model_feature_count != len(feature_names):
        raise ValueError(
            "\nXGBOOST MODEL / DATASET MISMATCH\n"
            f"Saved model features: {model_feature_count}\n"
            f"Data V3 features: {len(feature_names)}\n"
            "The saved XGBoost model must be trained "
            "using this Data V3 feature representation."
        )

    stored_names = booster.feature_names

    if stored_names is None:
        raise ValueError(
            "The saved model has no feature names. "
            "Cannot verify the training feature order."
        )

    if list(stored_names) != feature_names:
        raise ValueError(
            "Feature names or order differ between "
            "the saved model and Data V3."
        )

    # --------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------

    print("Loading Data V3 test dataset...")

    test_path = DATA_DIR / "known_test.parquet"

    test_data = pd.read_parquet(
        test_path,
        columns=feature_names + ["Label"]
    )

    X_test = test_data[feature_names]
    y_test = test_data["Label"]

    # --------------------------------------------------
    # LOAD LABEL MAPPING
    # --------------------------------------------------

    if hasattr(saved_model, "label_encoder"):
        label_classes = list(
            saved_model.label_encoder.classes_
        )

    else:
        train_path = DATA_DIR / "known_train.parquet"

        train_labels = pd.read_parquet(
            train_path,
            columns=["Label"]
        )

        encoder = LabelEncoder()
        encoder.fit(train_labels["Label"])

        label_classes = list(encoder.classes_)

    if len(label_classes) != model.n_classes_:
        raise ValueError(
            "\nCLASS COUNT MISMATCH\n"
            f"Model classes: {model.n_classes_}\n"
            f"Data V3 classes: {len(label_classes)}\n"
            "The saved model and label mapping "
            "must correspond to the same training run."
        )

    print("\n========== SHAP INPUT CHECK ==========")
    print("Model features :", model_feature_count)
    print("Model classes  :", model.n_classes_)
    print("Test samples   :", len(X_test))
    print("Feature match  : PASSED")
    print("======================================")

    return (
        model,
        X_test,
        y_test,
        feature_names,
        label_classes
    )
