import os
import sys

import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


from src.analysis.explainability import (
    XGBoostExplainer
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved-models",
    "xgboost_final.joblib"
)

TRAIN_PATH = os.path.join(
    PROJECT_ROOT,
    "notebooks",
    "Data_V1",
    "classifier_train.csv"
)

TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "notebooks",
    "Data_V1",
    "classifier_test.csv"
)

RESULT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "results",
    "shap_global_importance.csv"
)


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

def load_label_encoder():

    labels = pd.read_csv(
        TRAIN_PATH,
        usecols=["Label"]
    )

    encoder = LabelEncoder()

    encoder.fit(
        labels["Label"]
    )

    return encoder


# ============================================================
# LOAD TEST FEATURES
# ============================================================

def load_test_features():

    test = pd.read_csv(
        TEST_PATH
    )

    columns_to_drop = [
        column
        for column in [
            "Label",
            "source_file"
        ]
        if column in test.columns
    ]

    return test.drop(
        columns=columns_to_drop
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading trained XGBoost model..."
    )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Loading test data..."
    )

    X_test = load_test_features()

    label_encoder = load_label_encoder()

    booster_feature_names = (
        model.get_booster().feature_names
    )

    if booster_feature_names is None:
        feature_names = list(
            X_test.columns
        )

    else:
        feature_names = list(
            booster_feature_names
        )

    if set(X_test.columns) != set(feature_names):
        raise ValueError(
            "Test features do not match model features."
        )

    X_test = X_test[
        feature_names
    ]

    # --------------------------------------------------------
    # SAMPLE TEST DATA
    # --------------------------------------------------------

    # Global SHAP does not need all 377,603 samples.
    # A representative subset keeps computation manageable.
    sample_size = min(
        1000,
        len(X_test)
    )

    global_sample = X_test.sample(
        n=sample_size,
        random_state=42
    )

    print(
        f"Global SHAP sample size: {sample_size:,}"
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    explainer = XGBoostExplainer(
        model=model,
        feature_names=feature_names,
        label_classes=label_encoder.classes_,
        top_n=5
    )

    print(
        "Calculating global SHAP importance..."
    )

    importance = explainer.explain_global(
        data=global_sample,
        top_n=15
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        "\n========== GLOBAL SHAP IMPORTANCE =========="
    )

    print(
        importance.to_string(
            index=False
        )
    )

    print(
        "\n============================================"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            RESULT_PATH
        ),
        exist_ok=True
    )

    importance.to_csv(
        RESULT_PATH,
        index=False
    )

    print(
        f"\nSaved global SHAP results to:\n"
        f"{RESULT_PATH}"
    )


if __name__ == "__main__":
    main()