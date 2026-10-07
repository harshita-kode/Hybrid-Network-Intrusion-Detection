import os
import sys

import joblib
import numpy as np
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


# ============================================================
# LABEL MAPPING
# ============================================================

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


# ============================================================
# TEST DATA
# ============================================================

def load_test_data():

    test = pd.read_csv(
        TEST_PATH
    )

    if "Label" not in test.columns:
        raise ValueError(
            "classifier_test.csv does not contain Label."
        )

    columns_to_drop = [
        column
        for column in [
            "Label",
            "source_file"
        ]
        if column in test.columns
    ]

    X_test = test.drop(
        columns=columns_to_drop
    )

    y_test = test["Label"]

    return X_test, y_test


# ============================================================
# FIND EXAMPLE ATTACK
# ============================================================

def find_correct_attack_sample(
    model,
    X_test,
    y_test,
    label_encoder,
    search_size=5000
):

    search_size = min(
        search_size,
        len(X_test)
    )

    X_search = X_test.iloc[
        :search_size
    ]

    predictions = model.predict(
        X_search
    ).astype(int)

    for index, predicted_index in enumerate(
        predictions
    ):

        actual_class = str(
            y_test.iloc[index]
        )

        predicted_class = str(
            label_encoder.inverse_transform(
                [predicted_index]
            )[0]
        )

        if (
            actual_class != "BENIGN"
            and actual_class == predicted_class
        ):
            return index

    raise ValueError(
        "No correctly classified attack was found "
        f"within the first {search_size} test samples."
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
        "Loading label mapping..."
    )

    label_encoder = load_label_encoder()

    print(
        "Loading test data..."
    )

    X_test, y_test = load_test_data()

    # --------------------------------------------------------
    # MODEL FEATURES
    # --------------------------------------------------------

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
            "Test features do not match "
            "the trained XGBoost model."
        )

    # Exact training order.
    X_test = X_test[
        feature_names
    ]

    print(
        f"\nTest samples : {len(X_test):,}"
    )

    print(
        f"Features     : {len(feature_names)}"
    )

    print(
        f"Classes      : {len(label_encoder.classes_)}"
    )

    # --------------------------------------------------------
    # FIND A REAL CORRECTLY CLASSIFIED ATTACK
    # --------------------------------------------------------

    sample_index = find_correct_attack_sample(
        model=model,
        X_test=X_test,
        y_test=y_test,
        label_encoder=label_encoder
    )

    sample = X_test.iloc[
        [sample_index]
    ]

    actual_class = str(
        y_test.iloc[
            sample_index
        ]
    )

    print(
        f"\nSelected test index : {sample_index}"
    )

    print(
        f"Actual class        : {actual_class}"
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    print(
        "\nCreating SHAP explanation..."
    )

    explainer = XGBoostExplainer(
        model=model,
        feature_names=feature_names,
        label_classes=label_encoder.classes_,
        top_n=5
    )

    (
        predicted_class,
        confidence,
        explanation
    ) = explainer.explain(
        sample
    )

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    print(
        "\n========== SHAP EXPLANATION =========="
    )

    print(
        f"Actual Class     : {actual_class}"
    )

    print(
        f"Predicted Class  : {predicted_class}"
    )

    print(
        f"Confidence       : {confidence:.2%}"
    )

    print(
        "\nTop contributing features:"
    )

    for rank, contribution in enumerate(
        explanation.contributions,
        start=1
    ):

        print(
            f"\n{rank}. "
            f"{contribution.feature}"
        )

        print(
            "   Feature Value : "
            f"{contribution.feature_value:.6f}"
        )

        print(
            "   SHAP Value    : "
            f"{contribution.shap_value:+.6f}"
        )

    print(
        "\n======================================"
    )


if __name__ == "__main__":
    main()