
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

# Required because the saved XGBoost wrapper was serialized
# using the module path models.xgboost_classifier.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


MODEL_PATH = (
    PROJECT_ROOT
    / "saved-models"
    / "xgboost_final.joblib"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Data_V3_SSH_Unknown_final"
)

TEST_PATH = DATA_DIR / "known_test.parquet"


# ============================================================
# LOAD AND VALIDATE SHAP INPUTS
# ============================================================

def load_shap_inputs():
    """
    Load the trained XGBoost model and compatible test data.

    Uses the exact features stored in the trained classifier,
    rather than assuming the feature_list.json feature set.

    Returns
    -------
    model:
        Underlying trained XGBClassifier.

    X_test:
        Test features in the exact training feature order.

    y_test:
        Actual test labels.

    feature_names:
        Feature names used during training.

    label_classes:
        Original class labels from the saved LabelEncoder.
    """

    # --------------------------------------------------------
    # CHECK REQUIRED FILES
    # --------------------------------------------------------

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"XGBoost model not found: {MODEL_PATH}"
        )

    if not TEST_PATH.is_file():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}"
        )

    # --------------------------------------------------------
    # LOAD SAVED MODEL
    # --------------------------------------------------------

    print("Loading trained XGBoost model...")

    saved_model = joblib.load(MODEL_PATH)

    print(
        "Saved model type:",
        type(saved_model).__name__
    )

    # The current Data V3 model was saved as the custom
    # XGBoostIntrusionClassifier wrapper.
    if not hasattr(saved_model, "model"):
        raise TypeError(
            "Expected XGBoostIntrusionClassifier wrapper. "
            f"Received: {type(saved_model).__name__}"
        )

    if not hasattr(saved_model, "feature_names"):
        raise ValueError(
            "Saved classifier has no feature_names."
        )

    if not hasattr(saved_model, "label_encoder"):
        raise ValueError(
            "Saved classifier has no label_encoder."
        )

    # Extract the actual underlying XGBoost model.
    model = saved_model.model

    # --------------------------------------------------------
    # GET EXACT TRAINING FEATURES
    # --------------------------------------------------------

    if saved_model.feature_names is None:
        raise ValueError(
            "No training feature names stored in model."
        )

    feature_names = list(saved_model.feature_names)

    if not feature_names:
        raise ValueError(
            "Saved model feature list is empty."
        )

    if len(set(feature_names)) != len(feature_names):
        raise ValueError(
            "Duplicate feature names in saved model."
        )

    # --------------------------------------------------------
    # VERIFY XGBOOST MODEL FEATURES
    # --------------------------------------------------------

    booster = model.get_booster()

    model_feature_count = booster.num_features()

    if model_feature_count != len(feature_names):
        raise ValueError(
            "XGBoost feature count mismatch. "
            f"Booster expects {model_feature_count}, "
            f"but wrapper contains {len(feature_names)}."
        )

    booster_feature_names = booster.feature_names

    if booster_feature_names is not None:
        if list(booster_feature_names) != feature_names:
            raise ValueError(
                "Booster feature names/order do not match "
                "the saved classifier feature names."
            )

    # --------------------------------------------------------
    # GET CLASS LABELS
    # --------------------------------------------------------

    label_encoder = saved_model.label_encoder

    if not hasattr(label_encoder, "classes_"):
        raise ValueError(
            "Saved label encoder has not been fitted."
        )

    label_classes = list(label_encoder.classes_)

    if len(label_classes) != model.n_classes_:
        raise ValueError(
            "Class count mismatch. "
            f"XGBoost has {model.n_classes_} classes, "
            f"but LabelEncoder has {len(label_classes)}."
        )

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    print("Loading Data V3 known-test dataset...")

    # Load only the columns actually used by XGBoost,
    # plus the true label for evaluation.
    required_columns = feature_names + ["Label"]

    try:
        test_data = pd.read_parquet(
            TEST_PATH,
            columns=required_columns
        )

    except Exception as error:
        raise ValueError(
            "Unable to load the required model features "
            "from known_test.parquet. Check that the "
            "dataset matches the trained classifier."
        ) from error

    # --------------------------------------------------------
    # PREPARE FEATURES AND LABELS
    # --------------------------------------------------------

    X_test = test_data[feature_names].copy()

    y_test = test_data["Label"].copy()

    # Verify all features are numeric.
    non_numeric = X_test.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            "Non-numeric model features found: "
            f"{non_numeric}"
        )

    # Check for NaN and infinite values.
    values = X_test.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            "Test features contain NaN or infinite values."
        )

    if len(X_test) == 0:
        raise ValueError(
            "Test dataset is empty."
        )

    # --------------------------------------------------------
    # VERIFY FINAL FEATURE ORDER
    # --------------------------------------------------------

    if list(X_test.columns) != feature_names:
        raise ValueError(
            "Test feature order does not match "
            "the trained XGBoost model."
        )

    # --------------------------------------------------------
    # DISPLAY INPUT SUMMARY
    # --------------------------------------------------------

    print("\n========== SHAP INPUT CHECK ==========")

    print(
        "Model type       :",
        type(model).__name__
    )

    print(
        "Number of features:",
        len(feature_names)
    )

    print(
        "Number of classes :",
        len(label_classes)
    )

    print(
        "Test samples      :",
        f"{len(X_test):,}"
    )

    print(
        "Feature match     : PASSED"
    )

    print(
        "Class mapping     : PASSED"
    )

    print("\nModel classes:")

    for index, class_name in enumerate(label_classes):
        print(
            f"  {index}: {class_name}"
        )

    # Warn about potentially problematic features.
    suspicious_features = [
        feature
        for feature in [
            "id",
            "Attempted Category",
            "is_attempted"
        ]
        if feature in feature_names
    ]

    if suspicious_features:
        print(
            "\nWARNING: The model includes features "
            "that should be reviewed for possible "
            "identifier or label leakage:"
        )

        for feature in suspicious_features:
            print(
                f"  - {feature}"
            )

        print(
            "SHAP will explain the current model, "
            "but results should remain provisional "
            "until these features are reviewed."
        )

    print("\n======================================")

    # --------------------------------------------------------
    # RETURN SHAP INPUTS
    # --------------------------------------------------------

    return (
        model,
        X_test,
        y_test,
        feature_names,
        label_classes
    )
